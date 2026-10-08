// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Hyperlight Authors.

//! Which scratch pages the guest wrote since the last restore.
//!
//! A restore zeroes scratch. Zeroing all of it costs time in its size,
//! however little the guest wrote, and mapping fresh memory instead
//! costs a fault on every page the guest then touches. Where the
//! hypervisor logs the guest's writes, a restore zeroes only the pages
//! in the log (see [`HostSharedMemory::zero_written`]).
//!
//! On WHP, scratch is mapped with tracking, which costs the guest
//! nothing measurable, so the log is always used. Scratch is no longer
//! replaced on each restore, so the pages the guest writes stay
//! committed between restores, as all of scratch does on MSHV:
//! releasing them would cost a fault on each the next run touches.
//!
//! On MSHV, tracking is switched on for the whole VM. While it is on,
//! the guest's first write to each page after a read faults to the
//! hypervisor, and reading the log costs time in the size of scratch.
//! When reading the log, the faults and zeroing what was written cost
//! more than zeroing all of scratch, as when the guest writes most of it
//! or scratch is small, tracking is switched off, and retried later.
//! Each is measured as restores run, except the fault, which is
//! [`WRITE_FAULT`]. Each restore that zeroes all of scratch is timed and
//! the fastest kept, since a slower one was cold or interrupted;
//! [`WARM_RESETS`] gives warm ones before tracking first starts.
//!
//! [`HostSharedMemory::zero_written`]: crate::mem::shared_mem::HostSharedMemory::zero_written

use std::time::{Duration, Instant};

use tracing::{debug, warn};

use crate::hypervisor::virtual_machine::{DirtyLog, DirtyTracking, HypervisorError};
use crate::mem::mgr::ScratchZeroed;
use crate::mem::shared_mem::{DIRTY_PAGE_SIZE, DirtyRuns};

/// What tracking costs the guest per page it writes: the fault on its
/// first write after a read. Measured on MSHV nested in an Azure VM as
/// the difference per written page between tracked and untracked
/// restores (so it also holds the zeroing, counted again from
/// [`ScratchZeroed::Written`]); without nesting the fault costs less,
/// so this errs toward zeroing all.
const WRITE_FAULT: Duration = Duration::from_micros(1);

/// Restores in a row tracking must cost more than zeroing all of
/// scratch before it is switched off.
const COSTLIER_RESETS: u32 = 3;

/// Restores tracking stays off before it is retried.
const OFF_RESETS: u32 = 256;

/// Restores that zero all of scratch before tracking first starts. The
/// first touches memory not yet backed, so it is not a fair measure.
const WARM_RESETS: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Not tracking; restores zero everything. Tracking starts after
    /// `left` more restores.
    Off { left: u32 },
    /// Tracking since the last restore. It cost more than zeroing all of
    /// scratch for the last `costlier` restores.
    On { costlier: u32 },
    /// The hypervisor failed; restores zero everything.
    Failed,
}

/// The guest's writes to scratch since the last restore, where the
/// hypervisor logs them. See the module docs.
#[derive(Debug)]
pub(crate) struct ScratchDirtyLog {
    state: State,
    /// The last log read, reused across restores where the hypervisor
    /// API fills a caller's buffer.
    bitmap: Vec<u64>,
    /// Tracking is switched on in the hypervisor.
    enabled: bool,
    /// The fastest restore that zeroed all of scratch, since tracking
    /// was last switched off.
    zero_all: Option<Duration>,
    /// How long the last restore took to zero what was written.
    zero_written: Duration,
    /// The scratch range, (gpa, size), these apply to. Another one
    /// starts over.
    range: (u64, usize),
}

impl Default for ScratchDirtyLog {
    fn default() -> Self {
        Self {
            state: State::Off { left: WARM_RESETS },
            bitmap: Vec::new(),
            enabled: false,
            zero_all: None,
            zero_written: Duration::ZERO,
            range: (0, 0),
        }
    }
}

impl ScratchDirtyLog {
    /// Called once per restore, before scratch at `[gpa, gpa + size)`
    /// is reset. Returns the pages the guest wrote since the last
    /// restore, one bit per 4 KiB page, or `None` when they are not
    /// known and all of scratch must be reset. Then the caller reports
    /// how it zeroed scratch with [`zeroed`](Self::zeroed).
    pub(crate) fn take(
        &mut self,
        vm: &mut (impl DirtyLog + ?Sized),
        gpa: u64,
        size: usize,
    ) -> Option<&mut Vec<u64>> {
        let result = match vm.dirty_tracking() {
            DirtyTracking::None => return None,
            _ if self.state == State::Failed => return None,
            DirtyTracking::Mapped => vm
                .read_dirty_log(gpa, size, &mut self.bitmap)
                .map(|()| true),
            DirtyTracking::Switched => self.switched(vm, gpa, size),
        };
        match result {
            Ok(true) => Some(&mut self.bitmap),
            Ok(false) => None,
            Err(e) => {
                warn!("Scratch dirty log failed, restores zero all of scratch from now on: {e}");
                self.state = State::Failed;
                // Best effort. Left on, tracking costs each page one fault
                // at most: pages fault only after a read clears them.
                if self.enabled && vm.disable_dirty_tracking(gpa, size).is_ok() {
                    self.enabled = false;
                }
                None
            }
        }
    }

    /// How the restore after [`take`](Self::take) zeroed scratch.
    pub(crate) fn zeroed(&mut self, zeroed: ScratchZeroed) {
        match zeroed {
            ScratchZeroed::All(took) => {
                self.zero_all = Some(self.zero_all.map_or(took, |fastest| fastest.min(took)));
            }
            // Kept for the next decision, unless tracking just stopped.
            ScratchZeroed::Written(took) if matches!(self.state, State::On { .. }) => {
                self.zero_written = took;
            }
            ScratchZeroed::Written(_) => {}
        }
    }

    /// Step the switched-tracking state; true when `self.bitmap` holds
    /// the pages written since the last restore.
    fn switched(
        &mut self,
        vm: &mut (impl DirtyLog + ?Sized),
        gpa: u64,
        size: usize,
    ) -> Result<bool, HypervisorError> {
        if (gpa, size) != self.range {
            // Scratch was replaced: what was measured is of another one.
            // Tracking stays on, since the old range is unmapped and
            // MSHV disables it only over ranges read; the new range was
            // never read, so it costs the guest nothing until it is.
            self.state = State::Off { left: WARM_RESETS };
            self.range = (gpa, size);
            self.zero_all = None;
            self.zero_written = Duration::ZERO;
        }
        match self.state {
            State::Failed => Ok(false),
            State::Off { left: left @ 1.. } => {
                self.state = State::Off { left: left - 1 };
                Ok(false)
            }
            State::Off { left: 0 } => {
                // This restore zeroes everything. The first read after
                // enabling may report every page, so it is read away now.
                if !self.enabled {
                    vm.enable_dirty_tracking()?;
                    self.enabled = true;
                }
                vm.read_dirty_log(gpa, size, &mut self.bitmap)?;
                self.state = State::On { costlier: 0 };
                Ok(false)
            }
            State::On { costlier } => {
                let start = Instant::now();
                vm.read_dirty_log(gpa, size, &mut self.bitmap)?;
                let pages = written(&self.bitmap, size / DIRTY_PAGE_SIZE);
                let cost = start.elapsed()
                    + self.zero_written
                    + WRITE_FAULT * u32::try_from(pages).unwrap_or(u32::MAX);
                let costlier = match self.zero_all {
                    Some(zero_all) if cost > zero_all => costlier + 1,
                    _ => 0,
                };
                self.state = if costlier >= COSTLIER_RESETS {
                    debug!(
                        "Scratch dirty tracking off for {OFF_RESETS} restores: {cost:?} a restore, \
                         zeroing all {:?}",
                        self.zero_all
                    );
                    vm.disable_dirty_tracking(gpa, size)?;
                    self.enabled = false;
                    self.zero_all = None;
                    self.zero_written = Duration::ZERO;
                    State::Off { left: OFF_RESETS }
                } else {
                    State::On { costlier }
                };
                Ok(true)
            }
        }
    }
}

/// The set bits below `pages` in `bitmap`.
fn written(bitmap: &[u64], pages: usize) -> usize {
    DirtyRuns::new(bitmap, pages).map(|run| run.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPA: u64 = 0x1_0000_0000;
    const SIZE: usize = 16 << 20;
    const PAGES: usize = SIZE / 4096;

    /// A VM whose guest writes the first `written` pages between reads.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct FakeVm {
        tracking: Option<DirtyTracking>,
        written: usize,
        fail: bool,
        enables: u32,
        disables: u32,
        reads: u32,
        /// The range the last disable was given.
        disabled: Option<(u64, usize)>,
    }

    impl DirtyLog for FakeVm {
        fn dirty_tracking(&self) -> DirtyTracking {
            self.tracking.unwrap_or(DirtyTracking::None)
        }
        fn enable_dirty_tracking(&mut self) -> Result<(), HypervisorError> {
            self.enables += 1;
            Ok(())
        }
        fn disable_dirty_tracking(&mut self, gpa: u64, size: usize) -> Result<(), HypervisorError> {
            self.disables += 1;
            self.disabled = Some((gpa, size));
            Ok(())
        }
        fn read_dirty_log(
            &mut self,
            gpa: u64,
            size: usize,
            bitmap: &mut Vec<u64>,
        ) -> Result<(), HypervisorError> {
            assert_eq!((gpa, size % 4096), (GPA, 0));
            if self.fail {
                return Err(HypervisorError::Injected);
            }
            self.reads += 1;
            *bitmap = vec![0; (size / 4096).div_ceil(64)];
            for page in 0..self.written {
                bitmap[page / 64] |= 1 << (page % 64);
            }
            Ok(())
        }
    }

    fn vm(tracking: DirtyTracking, written: usize) -> FakeVm {
        FakeVm {
            tracking: Some(tracking),
            written,
            ..FakeVm::default()
        }
    }

    /// A restore as the sandbox does one: zeroing all of scratch in
    /// `zero_all` when the log does not say what was written.
    fn restore(log: &mut ScratchDirtyLog, vm: &mut FakeVm, zero_all: Duration) -> Option<usize> {
        match log.take(vm, GPA, SIZE) {
            Some(bitmap) => {
                let pages = written(bitmap, PAGES);
                log.zeroed(ScratchZeroed::Written(Duration::ZERO));
                Some(pages)
            }
            None => {
                log.zeroed(ScratchZeroed::All(zero_all));
                None
            }
        }
    }

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn written_counts_only_pages_in_range() {
        assert_eq!(written(&[u64::MAX, u64::MAX], 70), 70);
        assert_eq!(written(&[0b1011], 3), 2);
        assert_eq!(written(&[], 10), 0);
    }

    #[test]
    fn untracked_vms_have_no_log() {
        let mut vm = vm(DirtyTracking::None, 1);
        let mut log = ScratchDirtyLog::default();
        assert_eq!(restore(&mut log, &mut vm, MS), None);
        assert_eq!((vm.enables, vm.disables, vm.reads), (0, 0, 0));
    }

    #[test]
    fn mapped_tracking_is_always_read() {
        let mut vm = vm(DirtyTracking::Mapped, PAGES);
        let mut log = ScratchDirtyLog::default();
        for _ in 0..10 {
            assert_eq!(restore(&mut log, &mut vm, MS), Some(PAGES));
        }
        assert_eq!((vm.enables, vm.disables, vm.reads), (0, 0, 10));
    }

    /// Restores before tracking first starts, and the one starting it.
    fn start(log: &mut ScratchDirtyLog, vm: &mut FakeVm) {
        for _ in 0..WARM_RESETS {
            assert_eq!(restore(log, vm, MS), None);
        }
        assert_eq!(vm.enables, 0);
        // Enabled and read away: everything is zeroed this time.
        assert_eq!(restore(log, vm, MS), None);
        assert_eq!((vm.enables, vm.reads), (1, 1));
    }

    #[test]
    fn switched_tracking_starts_after_warming_up() {
        let mut vm = vm(DirtyTracking::Switched, 1);
        let mut log = ScratchDirtyLog::default();
        start(&mut log, &mut vm);
        for _ in 0..1000 {
            assert_eq!(restore(&mut log, &mut vm, MS), Some(1));
        }
        assert_eq!(vm.disables, 0);
    }

    #[test]
    fn costlier_tracking_is_switched_off_then_retried() {
        // 2000 faults cost about 2 ms, more than zeroing all in 1 ms.
        let mut vm = vm(DirtyTracking::Switched, 2000);
        let mut log = ScratchDirtyLog::default();
        start(&mut log, &mut vm);
        for _ in 0..COSTLIER_RESETS {
            assert_eq!(restore(&mut log, &mut vm, MS), Some(2000));
        }
        assert_eq!(vm.disables, 1);
        for _ in 0..OFF_RESETS {
            assert_eq!(restore(&mut log, &mut vm, MS), None);
        }
        assert_eq!(vm.enables, 1);
        // Retried.
        assert_eq!(restore(&mut log, &mut vm, MS), None);
        assert_eq!(vm.enables, 2);
        assert_eq!(restore(&mut log, &mut vm, MS), Some(2000));
    }

    #[test]
    fn zeroing_what_was_written_counts() {
        // One page written, but zeroing what was written (as when the host
        // wrote much of scratch) takes longer than zeroing all.
        let mut vm = vm(DirtyTracking::Switched, 1);
        let mut log = ScratchDirtyLog::default();
        start(&mut log, &mut vm);
        for _ in 0..COSTLIER_RESETS + 1 {
            if log.take(&mut vm, GPA, SIZE).is_some() {
                log.zeroed(ScratchZeroed::Written(2 * MS));
            }
        }
        assert_eq!(vm.disables, 1);
        // Not counted against tracking when it is retried.
        assert_eq!((log.zero_all, log.zero_written), (None, Duration::ZERO));
    }

    #[test]
    fn tracking_stays_on_where_zeroing_all_costs_more() {
        // The same writes, but zeroing all of a larger scratch costs 5 ms.
        let mut vm = vm(DirtyTracking::Switched, 2000);
        let mut log = ScratchDirtyLog::default();
        for _ in 0..100 {
            restore(&mut log, &mut vm, 5 * MS);
        }
        assert_eq!(vm.disables, 0);
    }

    #[test]
    fn a_cheaper_restore_resets_the_count() {
        let mut vm = vm(DirtyTracking::Switched, 2000);
        let mut log = ScratchDirtyLog::default();
        start(&mut log, &mut vm);
        for _ in 0..COSTLIER_RESETS - 1 {
            restore(&mut log, &mut vm, MS);
        }
        vm.written = 1;
        restore(&mut log, &mut vm, MS);
        vm.written = 2000;
        for _ in 0..COSTLIER_RESETS - 1 {
            assert!(restore(&mut log, &mut vm, MS).is_some());
        }
        assert_eq!(vm.disables, 0);
    }

    #[test]
    fn a_failed_read_zeroes_everything_from_then_on() {
        let mut vm = vm(DirtyTracking::Switched, 1);
        let mut log = ScratchDirtyLog::default();
        start(&mut log, &mut vm);
        vm.fail = true;
        assert_eq!(restore(&mut log, &mut vm, MS), None);
        assert_eq!(vm.disables, 1);
        vm.fail = false;
        for _ in 0..OFF_RESETS + 2 {
            assert_eq!(restore(&mut log, &mut vm, MS), None);
        }
        assert_eq!((vm.enables, vm.reads), (1, 1));
    }

    #[test]
    fn a_failed_mapped_read_is_not_retried() {
        let mut vm = vm(DirtyTracking::Mapped, 1);
        let mut log = ScratchDirtyLog::default();
        vm.fail = true;
        assert_eq!(restore(&mut log, &mut vm, MS), None);
        vm.fail = false;
        assert_eq!(restore(&mut log, &mut vm, MS), None);
        assert_eq!(vm.reads, 0);
    }

    #[test]
    fn a_cold_first_zeroing_does_not_count() {
        // The first restores zero cold memory slowly, and look costlier
        // than tracking, which costs about 2 ms a restore here.
        let mut vm = vm(DirtyTracking::Switched, 2000);
        let mut log = ScratchDirtyLog::default();
        restore(&mut log, &mut vm, 10 * MS);
        for _ in 0..WARM_RESETS + COSTLIER_RESETS {
            restore(&mut log, &mut vm, MS);
        }
        assert_eq!(vm.disables, 1);
    }

    #[test]
    fn another_scratch_size_starts_over() {
        let mut vm = vm(DirtyTracking::Switched, 1);
        let mut log = ScratchDirtyLog::default();
        start(&mut log, &mut vm);
        assert_eq!(restore(&mut log, &mut vm, MS), Some(1));
        assert!(log.take(&mut vm, GPA, SIZE * 2).is_none());
        // Not disabled over the old range, which is unmapped, nor
        // enabled twice.
        assert_eq!((vm.disables, log.zero_all), (0, None));
        for _ in 0..WARM_RESETS {
            assert!(log.take(&mut vm, GPA, SIZE * 2).is_none());
        }
        assert_eq!(vm.enables, 1);
        assert!(log.take(&mut vm, GPA, SIZE * 2).is_some());
    }
}
