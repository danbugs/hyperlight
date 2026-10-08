// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Hyperlight Authors.

//! Resetting a sandbox's scratch region to zeros on restore, on KVM.
//!
//! Dropping a page (`MADV_DONTNEED`) makes the guest's next touch of it
//! fault twice, in the host for a zeroed page and in KVM to map it. A
//! guest touches a small, stable set of scratch pages per run. Zeroing
//! those in place keeps them mapped, which costs a fraction of the two
//! faults, so a reset zeroes the pages a run uses and drops the rest.
//!
//! `/proc/self/pagemap` reports each page's page-table entry. A page with
//! memory of its own is zeroed and kept. A page in swap, or mapped to
//! memory it shares (the zero page a read maps, or a page shared with a
//! forked child), is dropped. A page with an empty entry reads as zero
//! and is left alone. The kernel reports any page that holds data as
//! present or swapped, so this leaves no data behind. Pagemap is used
//! only after it reports a page this process wrote, read by the process
//! that opened it.

use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;

use tracing::{debug, warn};

use super::shared_mem::{ExclusiveSharedMemory, SharedMemory};

/// The most memory a reset keeps. A run that backs more is dropped every
/// reset, since zeroing it costs more than refaulting what the next run
/// touches.
const KEEP_RESIDENT_MAX: usize = super::RESIDENT_SCRATCH_MAX;

/// Full scans that learn how much a run backs and where, keeping the
/// most and all of the spans. More than one, so a reset after no run
/// does not set them.
const LEARN_RESETS: u32 = 3;

/// Kept pages fewer than this many pages apart share a span.
const SPAN_GAP_PAGES: usize = 256;

/// The most spans kept. The closest are merged past it. Each span costs
/// a pagemap read and a drop per reset.
const MAX_SPANS: usize = 16;

/// The most drops a scan makes inside its spans. A guest that scatters
/// its pages past it is dropped every reset (see [`Phase::TooBig`]).
const MAX_DROPS: usize = 64;

/// Resets between whole-region scans, and resets an oversized working
/// set is dropped for before it is learned again.
const RESETS_PER_SCAN: u32 = 64;

/// Pagemap entries read at a time.
const CHUNK_PAGES: usize = 512;

/// Pagemap entry bits (Documentation/admin-guide/mm/pagemap.rst).
const PM_PRESENT: u64 = 1 << 63;
const PM_SWAPPED: u64 = 1 << 62;
const PM_EXCLUSIVE: u64 = 1 << 56;

/// What a reset does about the region's pages.
#[derive(Debug)]
enum Phase {
    /// Nothing learned. The next reset drops everything. What the guest
    /// touched before the snapshot, booting say, is not a run's working
    /// set.
    Fresh,
    /// Learning from whole-region scans: the most bytes a run kept so
    /// far, and every span a run kept pages in.
    Learning {
        left: u32,
        baseline: usize,
        spans: Vec<Range<usize>>,
    },
    /// Keeping the pages a run backs inside `spans`, and dropping the
    /// rest. A whole-region scan runs when `until_scan` is 0, and its
    /// spans with the last one's (`scanned`) become `spans`, so spans a
    /// working set left behind go.
    Steady {
        baseline: usize,
        spans: Vec<Range<usize>>,
        scanned: Vec<Range<usize>>,
        until_scan: u32,
    },
    /// A run backed more than the limit, or scattered its pages past
    /// [`MAX_DROPS`], or pagemap failed. Every reset drops everything,
    /// `left` more times, then learning starts again.
    TooBig { left: u32 },
}

/// How a scratch region is reset in place. See the module docs.
#[derive(Debug)]
pub(crate) struct ScratchReset {
    /// The region (base, size) this applies to. Another one starts over.
    region: (usize, usize),
    phase: Phase,
    /// [`KEEP_RESIDENT_MAX`], lower in tests.
    limit: usize,
    pagemap: Option<Pagemap>,
    /// Pagemap could not be used, already logged.
    warned: bool,
    /// Pagemap fails, to test that.
    #[cfg(test)]
    fail_pagemap: bool,
}

impl Default for ScratchReset {
    fn default() -> Self {
        Self {
            region: (0, 0),
            phase: Phase::Fresh,
            limit: KEEP_RESIDENT_MAX,
            pagemap: None,
            warned: false,
            #[cfg(test)]
            fail_pagemap: false,
        }
    }
}

/// This process's `/proc/self/pagemap`.
#[derive(Debug)]
struct Pagemap {
    file: File,
    /// The process that opened it. A forked child reads its parent's
    /// pages through an inherited one.
    pid: u32,
}

impl Pagemap {
    /// Open pagemap, and check that it reports a page this process wrote
    /// as present and its own. Pagemap that cannot see that (an
    /// emulated `/proc`, say) is not used. A fork in between shares the
    /// page again, so the check is tried a few times.
    fn open() -> std::io::Result<Self> {
        let pagemap = Self {
            file: File::open("/proc/self/pagemap")?,
            pid: std::process::id(),
        };
        let mut probe = Box::new(0u64);
        for _ in 0..3 {
            // SAFETY: a valid, aligned `u64` of our own. Volatile, so
            // the write that backs the page happens.
            unsafe { std::ptr::write_volatile(&mut *probe, 1) };
            let mut entry = [0u64];
            pagemap.read(&*probe as *const u64 as usize, &mut entry)?;
            if kept(entry[0]) {
                return Ok(pagemap);
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "pagemap does not report this process's pages",
        ))
    }

    /// Read the entries of the pages from `addr` on into `entries`, at
    /// most [`CHUNK_PAGES`].
    fn read(&self, addr: usize, entries: &mut [u64]) -> std::io::Result<()> {
        // SAFETY: the bytes of `entries`, which any bit pattern fills.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(entries.as_mut_ptr().cast::<u8>(), entries.len() * 8)
        };
        let offset = (addr / page_size::get()) as u64 * 8;
        self.file.read_exact_at(bytes, offset)
    }
}

/// Backed by memory of its own: zero it and keep it.
fn kept(entry: u64) -> bool {
    entry & (PM_PRESENT | PM_EXCLUSIVE) == PM_PRESENT | PM_EXCLUSIVE
}

/// Holding data of its own or not: in swap, or mapped to shared memory.
/// Dropped when not kept.
fn held(entry: u64) -> bool {
    entry & (PM_PRESENT | PM_SWAPPED) != 0
}

/// The region a reset works on, borrowed exclusively.
struct Region<'a> {
    mem: &'a mut ExclusiveSharedMemory,
    pages: usize,
}

impl Region<'_> {
    fn addr(&self, page: usize) -> usize {
        self.mem.base_ptr() as usize + page * page_size::get()
    }

    /// Drop `pages`. They read as zero and refill on demand.
    fn drop_pages(&mut self, pages: Range<usize>) -> std::io::Result<()> {
        assert!(pages.start <= pages.end && pages.end <= self.pages);
        if pages.is_empty() {
            return Ok(());
        }
        // SAFETY: the pages lie in the region, a private anonymous
        // mapping borrowed exclusively.
        let ret = unsafe {
            libc::madvise(
                self.addr(pages.start) as *mut libc::c_void,
                pages.len() * page_size::get(),
                libc::MADV_DONTNEED,
            )
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }

    fn zero_pages(&mut self, pages: Range<usize>) {
        let page = page_size::get();
        self.mem.as_mut_slice()[pages.start * page..pages.end * page].fill(0);
    }

    fn hint_small_pages(&mut self) {
        // SAFETY: as in `drop_pages`. Failure leaves the default page
        // size, so it is ignored.
        unsafe {
            libc::madvise(
                self.mem.base_ptr() as *mut libc::c_void,
                self.mem.mem_size(),
                libc::MADV_NOHUGEPAGE,
            )
        };
    }
}

/// What a scan kept: bytes, and the kept pages merged into spans.
#[derive(Default)]
struct Scan {
    kept: usize,
    spans: Vec<Range<usize>>,
    drops: usize,
}

impl Scan {
    /// Kept more than `limit`, or dropped past [`MAX_DROPS`]: cheaper to
    /// drop it all.
    fn over(&self, limit: usize) -> bool {
        self.kept > limit || self.drops > MAX_DROPS
    }
}

impl ScratchReset {
    /// Reset `mem` to zeros, keeping or dropping pages as [`Phase`]
    /// describes. On an error the region may be partly reset, and the
    /// caller must reset it some other way.
    pub(crate) fn reset(&mut self, mem: &mut ExclusiveSharedMemory) -> std::io::Result<()> {
        let key = (mem.base_ptr() as usize, mem.mem_size());
        let pages = mem.mem_size() / page_size::get();
        let mut region = Region { mem, pages };
        if self.region != key {
            self.region = key;
            self.phase = Phase::Fresh;
        }
        let phase = std::mem::replace(&mut self.phase, Phase::Fresh);
        match self.step(&mut region, phase) {
            Ok(phase) => {
                self.phase = phase;
                Ok(())
            }
            Err(e) => {
                self.phase = too_big();
                Err(e)
            }
        }
    }

    fn step(&mut self, region: &mut Region<'_>, phase: Phase) -> std::io::Result<Phase> {
        let all = 0..region.pages;
        Ok(match phase {
            Phase::Fresh => {
                // The region is reset in place from here on. KVM maps it
                // 4 KiB at a time, since it is not 2 MiB aligned for the
                // guest, so a huge page only makes a touch, and a reset,
                // zero 2 MiB.
                region.hint_small_pages();
                region.drop_pages(all)?;
                learning()
            }
            Phase::TooBig { left } => {
                region.drop_pages(all)?;
                if left > 1 {
                    Phase::TooBig { left: left - 1 }
                } else {
                    learning()
                }
            }
            Phase::Learning {
                left,
                baseline,
                spans,
            } => {
                let scan = self.scan(region, std::slice::from_ref(&all))?;
                if scan.over(self.limit) {
                    region.drop_pages(all)?;
                    too_big()
                } else {
                    let baseline = baseline.max(scan.kept);
                    let spans = merge(&spans, &scan.spans);
                    if left > 1 {
                        Phase::Learning {
                            left: left - 1,
                            baseline,
                            spans,
                        }
                    } else {
                        Phase::Steady {
                            baseline,
                            scanned: spans.clone(),
                            spans,
                            until_scan: RESETS_PER_SCAN,
                        }
                    }
                }
            }
            Phase::Steady {
                baseline,
                spans,
                scanned,
                until_scan,
            } => {
                let full = until_scan == 0;
                let scan = if full {
                    self.scan(region, std::slice::from_ref(&all))?
                } else {
                    self.scan(region, &spans)?
                };
                if scan.over(self.limit) {
                    region.drop_pages(all)?;
                    too_big()
                } else if scan.kept > (2 * baseline).max(baseline + (1 << 20)) {
                    // One run backed far more than usual. Its pages would
                    // be zeroed on every reset, so learn again.
                    region.drop_pages(all)?;
                    learning()
                } else if full {
                    Phase::Steady {
                        baseline,
                        spans: merge(&scanned, &scan.spans),
                        scanned: scan.spans,
                        until_scan: RESETS_PER_SCAN,
                    }
                } else {
                    Phase::Steady {
                        baseline,
                        spans,
                        scanned,
                        until_scan: until_scan - 1,
                    }
                }
            }
        })
    }

    /// This process's pagemap, opened again after a fork.
    fn pagemap(&mut self) -> std::io::Result<&Pagemap> {
        if self
            .pagemap
            .as_ref()
            .is_none_or(|p| p.pid != std::process::id())
        {
            self.pagemap = None;
            #[cfg(test)]
            let opened = if self.fail_pagemap {
                Err(std::io::ErrorKind::Unsupported.into())
            } else {
                Pagemap::open()
            };
            #[cfg(not(test))]
            let opened = Pagemap::open();
            match opened {
                Ok(pagemap) => self.pagemap = Some(pagemap),
                Err(e) => {
                    if !self.warned {
                        warn!("scratch is dropped on every restore, pagemap is unusable: {e}");
                        self.warned = true;
                    }
                    return Err(e);
                }
            }
        }
        Ok(self.pagemap.as_ref().unwrap())
    }

    /// Zero the kept pages of `spans`, drop the rest of what is backed in
    /// them, and drop everything outside them. Stops once over the limit
    /// or the drops ([`Scan::over`]), for the caller to drop it all.
    fn scan(&mut self, region: &mut Region<'_>, spans: &[Range<usize>]) -> std::io::Result<Scan> {
        let mut scan = Scan::default();
        let mut entries = [0u64; CHUNK_PAGES];
        let limit = self.limit;
        let pagemap = self.pagemap()?;
        let mut next = 0;
        for span in spans {
            region.drop_pages(next..span.start)?;
            next = span.end;
            let mut chunk = span.start;
            while chunk < span.end {
                let n = CHUNK_PAGES.min(span.end - chunk);
                let addr = region.addr(chunk);
                pagemap.read(addr, &mut entries[..n])?;
                let mut i = 0;
                while i < n {
                    let keep = kept(entries[i]);
                    let start = i;
                    let mut drop = false;
                    while i < n && kept(entries[i]) == keep {
                        drop |= held(entries[i]);
                        i += 1;
                    }
                    let pages = chunk + start..chunk + i;
                    if !keep {
                        // One drop for the run between kept pages, empty
                        // entries and all, so the guest cannot make a
                        // reset drop page by page.
                        if drop {
                            region.drop_pages(pages)?;
                            scan.drops += 1;
                            if scan.over(limit) {
                                return Ok(scan);
                            }
                        }
                        continue;
                    }
                    region.zero_pages(pages.clone());
                    scan.kept += pages.len() * page_size::get();
                    if scan.over(limit) {
                        return Ok(scan);
                    }
                    match scan.spans.last_mut() {
                        Some(last) if pages.start - last.end < SPAN_GAP_PAGES => {
                            last.end = pages.end
                        }
                        _ => scan.spans.push(pages),
                    }
                }
                chunk += n;
            }
        }
        region.drop_pages(next..region.pages)?;
        Ok(scan)
    }
}

fn learning() -> Phase {
    Phase::Learning {
        left: LEARN_RESETS,
        baseline: 0,
        spans: Vec::new(),
    }
}

/// The spans covering both `a` and `b` (each sorted), merged as a scan
/// merges kept pages, then the closest merged until at most
/// [`MAX_SPANS`] are left.
fn merge(a: &[Range<usize>], b: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut all: Vec<Range<usize>> = a.iter().chain(b).cloned().collect();
    all.sort_by_key(|s| s.start);
    let mut out: Vec<Range<usize>> = Vec::with_capacity(all.len());
    for span in all {
        match out.last_mut() {
            Some(last) if span.start < last.end + SPAN_GAP_PAGES => {
                last.end = last.end.max(span.end)
            }
            _ => out.push(span),
        }
    }
    while out.len() > MAX_SPANS {
        let i = (0..out.len() - 1)
            .min_by_key(|&i| out[i + 1].start - out[i].end)
            .unwrap();
        out[i].end = out[i + 1].end;
        out.remove(i + 1);
    }
    out
}

fn too_big() -> Phase {
    debug!("scratch is dropped on the next {RESETS_PER_SCAN} restores");
    Phase::TooBig {
        left: RESETS_PER_SCAN,
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn page() -> usize {
        page_size::get()
    }

    fn region(pages: usize) -> ExclusiveSharedMemory {
        ExclusiveSharedMemory::new(pages * page()).unwrap()
    }

    fn write(mem: &mut ExclusiveSharedMemory, page_index: usize, byte: u8) {
        mem.as_mut_slice()[page_index * page() + 7] = byte;
    }

    fn entry(mem: &ExclusiveSharedMemory, page_index: usize) -> u64 {
        let pagemap = Pagemap {
            file: File::open("/proc/self/pagemap").unwrap(),
            pid: std::process::id(),
        };
        let mut e = [0u64];
        pagemap
            .read(mem.base_ptr() as usize + page_index * page(), &mut e)
            .unwrap();
        e[0]
    }

    /// Reading maps the zero page on dropped pages, which the next reset
    /// drops again.
    fn all_zero(mem: &mut ExclusiveSharedMemory) -> bool {
        mem.as_mut_slice().iter().all(|b| *b == 0)
    }

    fn reset(state: &mut ScratchReset, mem: &mut ExclusiveSharedMemory) {
        state.reset(mem).unwrap();
    }

    /// Every reset leaves the region zero, whatever the run touched: the
    /// usual working set, pages far outside the learned spans, and
    /// nothing at all.
    #[test]
    fn every_reset_leaves_the_region_zero() {
        let mut mem = region(2048);
        let mut state = ScratchReset::default();
        for round in 0..300usize {
            let far = 1500 + round % 300;
            match round % 7 {
                3 => {}
                5 => {
                    write(&mut mem, far, 0xa5);
                    write(&mut mem, 40 + round % 8, 0x5a);
                }
                _ => {
                    for p in 32..96 {
                        write(&mut mem, p, round as u8 | 1);
                    }
                }
            }
            let spans_only =
                matches!(state.phase, Phase::Steady { until_scan, .. } if until_scan > 0);
            reset(&mut state, &mut mem);
            if spans_only {
                // Outside the learned spans: dropped, not kept.
                assert!(!kept(entry(&mem, far)), "round {round}");
            }
            assert!(all_zero(&mut mem), "round {round}");
        }
        assert!(matches!(state.phase, Phase::Steady { .. }));
    }

    #[test]
    fn pagemap_bits() {
        assert!(!kept(0) && !held(0));
        assert!(kept(PM_PRESENT | PM_EXCLUSIVE));
        assert!(!kept(PM_PRESENT) && held(PM_PRESENT));
        assert!(!kept(PM_SWAPPED) && held(PM_SWAPPED));
    }

    /// Shared pages among empty ones inside a span cost one drop, and
    /// end up dropped.
    #[test]
    fn shared_pages_in_a_span_are_dropped() {
        let mut mem = region(512);
        let mut state = ScratchReset::default();
        for _ in 0..5 {
            write(&mut mem, 10, 1);
            write(&mut mem, 200, 1);
            reset(&mut state, &mut mem);
        }
        for p in (11..200).step_by(2) {
            assert_eq!(mem.as_mut_slice()[p * page()], 0);
        }
        assert!(held(entry(&mem, 11)));
        reset(&mut state, &mut mem);
        assert!((11..200).all(|p| !held(entry(&mem, p))));
    }

    #[test]
    fn the_working_set_stays_backed() {
        let mut mem = region(2048);
        let mut state = ScratchReset::default();
        for _ in 0..10 {
            for p in 32..96 {
                write(&mut mem, p, 1);
            }
            reset(&mut state, &mut mem);
        }
        assert!((32..96).all(|p| kept(entry(&mem, p))));
        assert!(all_zero(&mut mem));
    }

    #[test]
    fn a_working_set_over_the_limit_is_dropped() {
        let mut mem = region(2048);
        let mut state = ScratchReset {
            limit: 16 * page(),
            ..ScratchReset::default()
        };
        for _ in 0..5 {
            for p in 0..64 {
                write(&mut mem, p, 1);
            }
            reset(&mut state, &mut mem);
            assert!(!kept(entry(&mem, 10)));
            assert!(all_zero(&mut mem));
        }
        assert!(matches!(state.phase, Phase::TooBig { .. }));
    }

    #[test]
    fn another_region_starts_over() {
        let mut big = region(2048);
        let mut state = ScratchReset::default();
        for _ in 0..6 {
            write(&mut big, 1900, 1);
            reset(&mut state, &mut big);
        }
        let mut small = region(256);
        for _ in 0..6 {
            write(&mut small, 200, 1);
            reset(&mut state, &mut small);
            assert!(all_zero(&mut small));
        }
        match &state.phase {
            Phase::Steady { spans, .. } => assert!(spans.iter().all(|s| s.end <= 256)),
            other => panic!("{other:?}"),
        }
    }

    /// A page in swap holds the guest's data. The reset drops it.
    #[test]
    fn a_swapped_page_is_dropped() {
        let mut mem = region(512);
        let mut state = ScratchReset::default();
        for _ in 0..5 {
            write(&mut mem, 10, 1);
            reset(&mut state, &mut mem);
        }
        write(&mut mem, 10, 0x77);
        // SAFETY: the test's own mapping.
        unsafe {
            libc::madvise(
                mem.base_ptr().add(10 * page()) as *mut libc::c_void,
                page(),
                libc::MADV_PAGEOUT,
            )
        };
        if entry(&mem, 10) & PM_SWAPPED == 0 {
            eprintln!("no swap: a_swapped_page_is_dropped skipped");
            return;
        }
        reset(&mut state, &mut mem);
        assert!(all_zero(&mut mem));
    }

    /// A forked child resets its own pages, not the ones its parent's
    /// pagemap shows: page 15, in a learned span and empty in the parent,
    /// is one the child wrote.
    #[test]
    fn a_forked_child_resets_its_own_pages() {
        let mut mem = region(512);
        let mut state = ScratchReset::default();
        for _ in 0..5 {
            write(&mut mem, 10, 1);
            write(&mut mem, 20, 1);
            reset(&mut state, &mut mem);
        }
        assert!(matches!(&state.phase, Phase::Steady { spans, .. } if spans[..] == [10..21]));
        // SAFETY: the child writes its copy of `mem`, resets it (opening
        // pagemap, which allocates), and exits without unwinding. The
        // allocator is fork safe, and the child takes no other lock.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                write(&mut mem, 15, 0x5a);
                state.reset(&mut mem).is_ok() && all_zero(&mut mem)
            }));
            // SAFETY: ends the child.
            unsafe { libc::_exit(if matches!(ok, Ok(true)) { 0 } else { 1 }) };
        }
        let mut status = 0;
        // SAFETY: waits for the child above.
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    }

    /// Without pagemap, every reset drops everything, and says so once.
    #[test]
    fn without_pagemap_resets_drop() {
        let mut mem = region(512);
        let mut state = ScratchReset {
            fail_pagemap: true,
            ..ScratchReset::default()
        };
        write(&mut mem, 10, 1);
        reset(&mut state, &mut mem);
        write(&mut mem, 10, 1);
        assert!(state.reset(&mut mem).is_err());
        assert!(state.warned);
        assert!(matches!(state.phase, Phase::TooBig { .. }));
        for _ in 0..10 {
            write(&mut mem, 10, 1);
            reset(&mut state, &mut mem);
            assert!(all_zero(&mut mem));
        }
    }

    /// Pages scattered so that a reset would drop one by one are dropped
    /// all at once.
    #[test]
    fn scattered_pages_are_dropped_wholesale() {
        let mut mem = region(2048);
        let mut state = ScratchReset::default();
        let scatter = |mem: &mut ExclusiveSharedMemory| {
            for p in (0..400).step_by(2) {
                write(mem, p, 1);
                assert_eq!(mem.as_mut_slice()[(p + 1) * page()], 0);
            }
        };
        for _ in 0..6 {
            scatter(&mut mem);
            reset(&mut state, &mut mem);
            assert!(all_zero(&mut mem));
        }
        assert!(matches!(state.phase, Phase::TooBig { .. }));
    }

    #[test]
    fn spans_merge_to_at_most_max_spans() {
        let gap = SPAN_GAP_PAGES;
        let many: Vec<Range<usize>> = (0..40).map(|i| i * 2 * gap..i * 2 * gap + 1).collect();
        let merged = merge(&many, &[]);
        assert_eq!(merged.len(), MAX_SPANS);
        assert_eq!(merged.first().unwrap().start, 0);
        assert_eq!(merged.last().unwrap().end, 39 * 2 * gap + 1);
    }

    #[test]
    fn spans_merge() {
        let gap = SPAN_GAP_PAGES;
        assert_eq!(merge(&[0..1], &[1..2]), vec![0..2]);
        assert_eq!(
            merge(&[0..1, 3 * gap..3 * gap + 1], &[gap..gap + 1]),
            vec![0..gap + 1, 3 * gap..3 * gap + 1]
        );
        assert_eq!(merge(&[], &[5..6]), vec![5..6]);
    }
}
