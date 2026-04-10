/*
Copyright 2025 The Hyperlight Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(feature = "i686-guest"))]
use hyperlight_common::layout::scratch_base_gpa;
use hyperlight_common::layout::scratch_base_gva;
#[cfg(not(feature = "i686-guest"))]
use hyperlight_common::vmem::{self, BasicMapping, CowMapping};
use hyperlight_common::vmem::{Mapping, MappingKind, PAGE_SIZE};
use tracing::{Span, instrument};

use crate::HyperlightError::MemoryRegionSizeMismatch;
use crate::Result;
use crate::hypervisor::regs::CommonSpecialRegisters;
use crate::mem::exe::LoadInfo;
use crate::mem::layout::SandboxMemoryLayout;
use crate::mem::memory_region::MemoryRegion;
#[cfg(not(feature = "i686-guest"))]
use crate::mem::mgr::GuestPageTableBuffer;
use crate::mem::mgr::SnapshotSharedMemory;
use crate::mem::shared_mem::{ReadonlySharedMemory, SharedMemory};
use crate::sandbox::SandboxConfiguration;
use crate::sandbox::uninitialized::{GuestBinary, GuestEnvironment};

pub(super) static SANDBOX_CONFIGURATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Presently, a snapshot can be of a preinitialised sandbox, which
/// still needs an initialise function called in order to determine
/// how to call into it, or of an already-properly-initialised sandbox
/// which can be immediately called into. This keeps track of the
/// difference.
///
/// TODO: this should not necessarily be around in the long term:
/// ideally we would just preinitialise earlier in the snapshot
/// creation process and never need this.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum NextAction {
    /// A sandbox in the preinitialise state still needs to be
    /// initialised by calling the initialise function
    Initialise(u64),
    /// A sandbox in the ready state can immediately be called into,
    /// using the dispatch function pointer.
    Call(u64),
    /// Only when compiling for tests: a sandbox that cannot actually
    /// be used
    #[cfg(test)]
    None,
}

/// A wrapper around a `SharedMemory` reference and a snapshot
/// of the memory therein
pub struct Snapshot {
    /// Unique ID of the sandbox configuration for sandboxes where
    /// this snapshot may be restored.
    sandbox_id: u64,
    /// Layout object for the sandbox. TODO: get rid of this and
    /// replace with something saner and set up from the guest (early
    /// on?).
    ///
    /// Not checked on restore, since any sandbox with the same
    /// configuration id will share the same layout
    layout: crate::mem::layout::SandboxMemoryLayout,
    /// Memory of the sandbox at the time this snapshot was taken
    memory: ReadonlySharedMemory,
    /// The memory regions that were mapped when this snapshot was
    /// taken (excluding initial sandbox regions)
    regions: Vec<MemoryRegion>,
    /// Separate PT storage for non-compacting i686 snapshots.
    /// Empty for compacting snapshots (PTs are in `memory`).
    separate_pt_bytes: Vec<u8>,
    /// Extra debug information about the binary in this snapshot,
    /// from when the binary was first loaded into the snapshot.
    ///
    /// This information is provided on a best-effort basis, and there
    /// is a pretty good chance that it does not exist; generally speaking,
    /// things like persisting a snapshot and reloading it are likely
    /// to destroy this information.
    load_info: LoadInfo,
    /// The hash of the other portions of the snapshot. Morally, this
    /// is just a memoization cache for [`hash`], below, but it is not
    /// a [`std::sync::OnceLock`] because it may be persisted to disk
    /// without being recomputed on load.
    ///
    /// It is not a [`blake3::Hash`] because we do not presently
    /// require constant-time equality checking
    hash: [u8; 32],
    /// The address of the top of the guest stack
    stack_top_gva: u64,

    /// Special register state captured from the vCPU during snapshot.
    /// None for snapshots created directly from a binary (before
    /// guest runs).  Some for snapshots taken from a running sandbox.
    /// Note: CR3 in this struct is NOT used on restore, since page
    /// tables are relocated during snapshot.
    sregs: Option<CommonSpecialRegisters>,

    /// The next action that should be performed on this snapshot
    entrypoint: NextAction,
}
impl core::convert::AsRef<Snapshot> for Snapshot {
    fn as_ref(&self) -> &Self {
        self
    }
}
impl hyperlight_common::vmem::TableReadOps for Snapshot {
    type TableAddr = u64;
    fn entry_addr(addr: u64, offset: u64) -> u64 {
        addr + offset
    }
    unsafe fn read_entry(&self, addr: u64) -> u64 {
        let addr = addr as usize;
        let Some(pte_bytes) = self.memory.as_slice().get(addr..addr + 8) else {
            // Attacker-controlled data pointed out-of-bounds. We'll
            // default to returning 0 in this case, which, for most
            // architectures (including x86-64 and arm64, the ones we
            // care about presently) will be a not-present entry.
            return 0;
        };
        // this is statically the correct size, so using unwrap() here
        // doesn't make this any more panic-y.
        #[allow(clippy::unwrap_used)]
        let n: [u8; 8] = pte_bytes.try_into().unwrap();
        u64::from_ne_bytes(n)
    }
    fn to_phys(addr: u64) -> u64 {
        addr
    }
    fn from_phys(addr: u64) -> u64 {
        addr
    }
    fn root_table(&self) -> u64 {
        self.root_pt_gpa()
    }
}

/// Compute a deterministic hash of a snapshot.
///
/// This does not include the load info from the snapshot, because
/// that is only used for debugging builds.
fn hash(memory: &[u8], regions: &[MemoryRegion]) -> Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(memory);
    for rgn in regions {
        hasher.update(&usize::to_le_bytes(rgn.guest_region.start));
        let guest_len = rgn.guest_region.end - rgn.guest_region.start;
        #[allow(clippy::useless_conversion)]
        let host_start_addr: usize = rgn.host_region.start.into();
        #[allow(clippy::useless_conversion)]
        let host_end_addr: usize = rgn.host_region.end.into();
        hasher.update(&usize::to_le_bytes(host_start_addr));
        let host_len = host_end_addr - host_start_addr;
        if guest_len != host_len {
            return Err(MemoryRegionSizeMismatch(
                host_len,
                guest_len,
                format!("{:?}", rgn),
            ));
        }
        // Ignore [`MemoryRegion::region_type`], since it is extra
        // information for debugging rather than a core part of the
        // identity of the snapshot/workload.
        hasher.update(&usize::to_le_bytes(guest_len));
        hasher.update(&u32::to_le_bytes(rgn.flags.bits()));
    }
    // Ignore [`load_info`], since it is extra information for
    // debugging rather than a core part of the identity of the
    // snapshot/workload.
    Ok(hasher.finalize().into())
}

pub(crate) fn access_gpa<'a>(
    snap: &'a [u8],
    scratch: &'a [u8],
    layout: SandboxMemoryLayout,
    gpa: u64,
) -> Option<(&'a [u8], usize)> {
    let resolved = layout.resolve_gpa(gpa, &[])?.with_memories(snap, scratch);
    Some((resolved.base.as_ref(), resolved.offset))
}

pub(crate) struct SharedMemoryPageTableBuffer<'a> {
    snap: &'a [u8],
    scratch: &'a [u8],
    layout: SandboxMemoryLayout,
    root: u64,
    /// CoW resolution map: maps snapshot GPAs to their CoW'd scratch GPAs.
    /// Built by walking the kernel PD to find pages that were CoW'd during boot.
    #[cfg(feature = "i686-guest")]
    cow_map: std::collections::HashMap<u64, u64>,
}
impl<'a> SharedMemoryPageTableBuffer<'a> {
    pub(crate) fn new(
        snap: &'a [u8],
        scratch: &'a [u8],
        layout: SandboxMemoryLayout,
        root: u64,
    ) -> Self {
        Self {
            snap,
            scratch,
            layout,
            root,
            #[cfg(feature = "i686-guest")]
            cow_map: std::collections::HashMap::new(),
        }
    }

}
impl<'a> hyperlight_common::vmem::TableReadOps for SharedMemoryPageTableBuffer<'a> {
    type TableAddr = u64;
    fn entry_addr(addr: u64, offset: u64) -> u64 {
        addr + offset
    }
    unsafe fn read_entry(&self, addr: u64) -> u64 {
        // For i686: if the GPA was CoW'd, read from the scratch copy instead.
        #[cfg(feature = "i686-guest")]
        let addr = {
            let page_gpa = addr & 0xFFFFF000;
            if let Some(&scratch_gpa) = self.cow_map.get(&page_gpa) {
                scratch_gpa + (addr & 0xFFF)
            } else {
                addr
            }
        };
        let memoff = access_gpa(self.snap, self.scratch, self.layout, addr);
        // For i686 guests, page table entries are 4 bytes; for x86_64 they
        // are 8 bytes. Read the correct size based on the feature flag.
        #[cfg(feature = "i686-guest")]
        {
            let Some(pte_bytes) = memoff.and_then(|(mem, off)| mem.get(off..off + 4)) else {
                return 0;
            };
            #[allow(clippy::unwrap_used)]
            let n: [u8; 4] = pte_bytes.try_into().unwrap();
            u32::from_ne_bytes(n) as u64
        }
        #[cfg(not(feature = "i686-guest"))]
        {
            let Some(pte_bytes) = memoff.and_then(|(mem, off)| mem.get(off..off + 8)) else {
                return 0;
            };
            #[allow(clippy::unwrap_used)]
            let n: [u8; 8] = pte_bytes.try_into().unwrap();
            u64::from_ne_bytes(n)
        }
    }
    fn to_phys(addr: u64) -> u64 {
        addr
    }
    fn from_phys(addr: u64) -> u64 {
        addr
    }
    fn root_table(&self) -> u64 {
        self.root
    }
}
impl<'a> core::convert::AsRef<SharedMemoryPageTableBuffer<'a>> for SharedMemoryPageTableBuffer<'a> {
    fn as_ref(&self) -> &Self {
        self
    }
}
/// Build a CoW resolution map by walking a kernel PD.
/// For each PTE that maps a VA in [0, MEMORY_SIZE) to a PA in scratch,
/// record: original_gpa → scratch_gpa.
#[cfg(feature = "i686-guest")]
fn build_cow_map(
    snap: &[u8],
    scratch: &[u8],
    layout: SandboxMemoryLayout,
    kernel_root: u64,
) -> std::collections::HashMap<u64, u64> {
    use hyperlight_common::layout::scratch_base_gpa;
    let mut cow_map = std::collections::HashMap::new();
    let scratch_base = scratch_base_gpa(layout.get_scratch_size());
    let scratch_end = scratch_base + layout.get_scratch_size() as u64;
    let mem_size = layout.get_memory_size().unwrap_or(0) as u64;

    for pdi in 0..1024u64 {
        let pde_addr = kernel_root + pdi * 4;
        let pde = access_gpa(snap, scratch, layout, pde_addr)
            .and_then(|(mem, off)| mem.get(off..off + 4))
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .unwrap_or(0);
        if (pde & 1) == 0 {
            continue;
        }
        let pt_gpa = (pde & 0xFFFFF000) as u64;
        for pti in 0..1024u64 {
            let pte_addr = pt_gpa + pti * 4;
            let pte = access_gpa(snap, scratch, layout, pte_addr)
                .and_then(|(mem, off)| mem.get(off..off + 4))
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .unwrap_or(0);
            if (pte & 1) == 0 {
                continue;
            }
            let frame_gpa = (pte & 0xFFFFF000) as u64;
            let va = (pdi << 22) | (pti << 12);
            if va < mem_size && frame_gpa >= scratch_base && frame_gpa < scratch_end {
                cow_map.insert(va, frame_gpa);
            }
        }
    }
    cow_map
}

fn filtered_mappings<'a>(
    snap: &'a [u8],
    scratch: &'a [u8],
    regions: &[MemoryRegion],
    layout: SandboxMemoryLayout,
    root_pts: &[u64],
    #[cfg(feature = "i686-guest")] cow_map: &std::collections::HashMap<u64, u64>,
) -> Vec<(Mapping, &'a [u8])> {
    #[cfg(not(feature = "i686-guest"))]
    let mappings_iter: Vec<Mapping> = {
        // For x86_64, only a single root is supported today.
        let root_pt = root_pts.first().copied().unwrap_or(0);
        let op = SharedMemoryPageTableBuffer::new(snap, scratch, layout, root_pt);
        unsafe {
            hyperlight_common::vmem::virt_to_phys(
                &op,
                0,
                hyperlight_common::layout::MAX_GVA as u64,
            )
        }
        .collect()
    };

    #[cfg(feature = "i686-guest")]
    let mappings_iter: Vec<Mapping> = {
        use std::collections::HashSet;
        let mut mappings = Vec::new();
        let mut seen_phys = HashSet::new();

        let scratch_base_gva_val = hyperlight_common::layout::scratch_base_gva(layout.get_scratch_size());
        for &root_pt in root_pts {
            let mut op = SharedMemoryPageTableBuffer::new(snap, scratch, layout, root_pt);
            op.cow_map = cow_map.clone();
            let root_mappings =
                unsafe { hyperlight_common::i686_guest_vmem::virt_to_phys_all(&op) };
            for m in root_mappings {
                // Skip mappings whose VA is in the scratch region — these
                // are identity-mapped helpers and would poison seen_phys for
                // legitimate user mappings that share the same scratch PAs.
                if m.virt_base >= scratch_base_gva_val {
                    continue;
                }
                if seen_phys.insert(m.phys_base) {
                    mappings.push(m);
                }
            }
        }
        mappings
    };

    mappings_iter
        .into_iter()
        .filter_map(move |mapping| {
            // the scratch map doesn't count
            if mapping.virt_base >= scratch_base_gva(layout.get_scratch_size()) {
                return None;
            }
            // neither does the mapping of the snapshot's own page tables
            #[cfg(not(feature = "i686-guest"))]
            if mapping.virt_base >= hyperlight_common::layout::SNAPSHOT_PT_GVA_MIN as u64
                && mapping.virt_base <= hyperlight_common::layout::SNAPSHOT_PT_GVA_MAX as u64
            {
                return None;
            }
            // todo: is it useful to warn if we can't resolve this?
            let contents =
                unsafe { guest_page(snap, scratch, regions, layout, mapping.phys_base) }?;
            Some((mapping, contents))
        })
        .collect()
}

/// Find the contents of the page which starts at gpa in guest physical
/// memory, taking into account excess host->guest regions
///
/// # Safety
/// The host side of the regions identified by MemoryRegion must be
/// alive and must not be mutated by any other thread: references to
/// these regions may be created and live for `'a`.
unsafe fn guest_page<'a>(
    snap: &'a [u8],
    scratch: &'a [u8],
    regions: &[MemoryRegion],
    layout: SandboxMemoryLayout,
    gpa: u64,
) -> Option<&'a [u8]> {
    let resolved = layout
        .resolve_gpa(gpa, regions)?
        .with_memories(snap, scratch);
    if resolved.as_ref().len() < PAGE_SIZE {
        return None;
    }
    Some(&resolved.as_ref()[..PAGE_SIZE])
}

#[cfg(not(feature = "i686-guest"))]
fn map_specials(pt_buf: &GuestPageTableBuffer, scratch_size: usize) {
    // Map the scratch region
    let mapping = Mapping {
        phys_base: scratch_base_gpa(scratch_size),
        virt_base: scratch_base_gva(scratch_size),
        len: scratch_size as u64,
        kind: MappingKind::Basic(BasicMapping {
            readable: true,
            writable: true,
            // assume that the guest will map these pages elsewhere if
            // it actually needs to execute from them
            executable: false,
        }),
    };
    unsafe { vmem::map(pt_buf, mapping) };
}

impl Snapshot {
    /// Create a new snapshot from the guest binary identified by `env`. With the configuration
    /// specified in `cfg`.
    pub(crate) fn from_env<'a, 'b>(
        env: impl Into<GuestEnvironment<'a, 'b>>,
        cfg: SandboxConfiguration,
    ) -> Result<Self> {
        let env = env.into();
        let mut bin = env.guest_binary;
        bin.canonicalize()?;
        let blob = env.init_data;

        use crate::mem::exe::ExeInfo;
        let exe_info = match bin {
            GuestBinary::FilePath(bin_path_str) => ExeInfo::from_file(&bin_path_str)?,
            GuestBinary::Buffer(buffer) => ExeInfo::from_buf(buffer)?,
        };

        // Check guest/host version compatibility.
        let host_version = env!("CARGO_PKG_VERSION");
        if let Some(v) = exe_info.guest_bin_version()
            && v != host_version
        {
            return Err(crate::HyperlightError::GuestBinVersionMismatch {
                guest_bin_version: v.to_string(),
                host_version: host_version.to_string(),
            });
        }

        let guest_blob_size = blob.as_ref().map(|b| b.data.len()).unwrap_or(0);
        let guest_blob_mem_flags = blob.as_ref().map(|b| b.permissions);

        let mut layout = crate::mem::layout::SandboxMemoryLayout::new(
            cfg,
            exe_info.loaded_size(),
            guest_blob_size,
            guest_blob_mem_flags,
        )?;

        let load_addr = layout.get_guest_code_address() as u64;
        let base_va = exe_info.base_va();
        let entrypoint_va: u64 = exe_info.entrypoint().into();

        let mut memory = vec![0; layout.get_memory_size()?];

        let load_info = exe_info.load(
            load_addr.try_into()?,
            &mut memory[layout.get_guest_code_offset()..],
        )?;

        layout.write_peb(&mut memory)?;

        blob.map(|x| layout.write_init_data(&mut memory, x.data))
            .transpose()?;

        #[cfg(not(feature = "i686-guest"))]
        {
            // Set up 4-level (amd64) page table entries for the snapshot
            let pt_buf = GuestPageTableBuffer::new(layout.get_pt_base_gpa() as usize);

            use crate::mem::memory_region::{GuestMemoryRegion, MemoryRegionFlags};

            for rgn in layout.get_memory_regions_::<GuestMemoryRegion>(())?.iter() {
                let readable = rgn.flags.contains(MemoryRegionFlags::READ);
                let executable = rgn.flags.contains(MemoryRegionFlags::EXECUTE);
                let writable = rgn.flags.contains(MemoryRegionFlags::WRITE);
                let kind = if writable {
                    MappingKind::Cow(CowMapping {
                        readable,
                        executable,
                    })
                } else {
                    MappingKind::Basic(BasicMapping {
                        readable,
                        writable: false,
                        executable,
                    })
                };
                let mapping = Mapping {
                    phys_base: rgn.guest_region.start as u64,
                    virt_base: rgn.guest_region.start as u64,
                    len: rgn.guest_region.len() as u64,
                    kind,
                };
                unsafe { vmem::map(&pt_buf, mapping) };
            }

            map_specials(&pt_buf, layout.get_scratch_size());

            let pt_bytes = pt_buf.into_bytes();
            layout.set_pt_size(pt_bytes.len())?;
            memory.extend(&pt_bytes);
        };
        #[cfg(feature = "i686-guest")]
        {
            // Build 2-level (i686) page tables directly with 4-byte PTEs.
            use crate::mem::memory_region::{GuestMemoryRegion, MemoryRegionFlags};

            let page_size: usize = 4096;
            // PDE entries MUST reference scratch GPAs (not shared memory) because
            // the CoW handler accesses PT pages via virtual addresses. The guest PT
            // maps scratch (writable), so VA accesses to PT pages in scratch work.
            // Using shared memory GPAs would require mapping the PT area in the
            // guest PT (self-referencing), which we don't do.
            let pd_base_gpa = layout.get_pt_base_gpa() as usize;
            let mut pt_bytes: Vec<u8> = vec![0u8; page_size]; // Start with PD (one 4KB page)

            // i686 PTE flags from arch-specific module.
            use hyperlight_common::i686_guest_vmem as i686_vmem;
            const PTE_PRESENT: u32 = i686_vmem::PAGE_PRESENT as u32;
            const PTE_RW: u32 = i686_vmem::PAGE_RW as u32;
            const PTE_ACCESSED: u32 = i686_vmem::PAGE_ACCESSED as u32;
            const PTE_COW: u32 = i686_vmem::PAGE_AVL_COW as u32;

            let mut map_region = |va_start: u64, pa_start: u64, len: u64, writable: bool| {
                let mut va = va_start;
                let mut pa = pa_start;
                let end = va_start + len;
                let leaf_flags: u32 = if writable {
                    // Guest-assisted CoW: writable pages are marked read-only
                    // with the CoW bit. The guest #PF handler allocates
                    // scratch pages on first write, keeping VA==PA identity
                    // mapping consistent for the frame allocator.
                    PTE_PRESENT | PTE_ACCESSED | PTE_COW
                } else {
                    PTE_PRESENT | PTE_ACCESSED // Genuinely read-only
                };

                while va < end {
                    let pdi = ((va as u32 >> 22) & 0x3FF) as usize;
                    let pti = ((va as u32 >> 12) & 0x3FF) as usize;

                    // Check if PDE exists
                    let pde_off = pdi * 4;
                    let pde = u32::from_le_bytes([
                        pt_bytes[pde_off],
                        pt_bytes[pde_off + 1],
                        pt_bytes[pde_off + 2],
                        pt_bytes[pde_off + 3],
                    ]);

                    let pt_offset_in_buf = if (pde & PTE_PRESENT) == 0 {
                        // Allocate a new PT page
                        let pt_offset = pt_bytes.len();
                        pt_bytes.resize(pt_offset + page_size, 0);
                        let pt_gpa = (pd_base_gpa + pt_offset) as u32;
                        let new_pde = pt_gpa | PTE_PRESENT | PTE_RW | PTE_ACCESSED;
                        pt_bytes[pde_off..pde_off + 4]
                            .copy_from_slice(&new_pde.to_le_bytes());
                        pt_offset
                    } else {
                        let pt_gpa = pde & 0xFFFFF000;
                        (pt_gpa as usize) - pd_base_gpa
                    };

                    // Write PTE
                    let pte_off = pt_offset_in_buf + pti * 4;
                    let pte = (pa as u32) | leaf_flags;
                    pt_bytes[pte_off..pte_off + 4]
                        .copy_from_slice(&pte.to_le_bytes());

                    va += page_size as u64;
                    pa += page_size as u64;
                }
            };

            // 1. Map snapshot memory regions
            for rgn in layout.get_memory_regions_::<GuestMemoryRegion>(())?.iter() {
                let writable = rgn.flags.contains(MemoryRegionFlags::WRITE);
                map_region(
                    rgn.guest_region.start as u64,
                    rgn.guest_region.start as u64,
                    rgn.guest_region.len() as u64,
                    writable,
                );
            }

            // 2. Map scratch region (always writable, not CoW)
            {
                let scratch_size = layout.get_scratch_size();
                let scratch_gpa = hyperlight_common::layout::scratch_base_gpa(scratch_size);
                let scratch_gva = hyperlight_common::layout::scratch_base_gva(scratch_size);
                let mut va = scratch_gva;
                let mut pa = scratch_gpa;
                let end = scratch_gva + scratch_size as u64;
                while va < end {
                    let pdi = ((va as u32 >> 22) & 0x3FF) as usize;
                    let pti = ((va as u32 >> 12) & 0x3FF) as usize;

                    let pde_off = pdi * 4;
                    let pde = u32::from_le_bytes([
                        pt_bytes[pde_off],
                        pt_bytes[pde_off + 1],
                        pt_bytes[pde_off + 2],
                        pt_bytes[pde_off + 3],
                    ]);

                    let pt_offset_in_buf = if (pde & PTE_PRESENT) == 0 {
                        let pt_offset = pt_bytes.len();
                        pt_bytes.resize(pt_offset + page_size, 0);
                        let pt_gpa = (pd_base_gpa + pt_offset) as u32;
                        let new_pde = pt_gpa | PTE_PRESENT | PTE_RW | PTE_ACCESSED;
                        pt_bytes[pde_off..pde_off + 4]
                            .copy_from_slice(&new_pde.to_le_bytes());
                        pt_offset
                    } else {
                        let pt_gpa = pde & 0xFFFFF000;
                        (pt_gpa as usize) - pd_base_gpa
                    };

                    let pte_off = pt_offset_in_buf + pti * 4;
                    let pte = (pa as u32) | PTE_PRESENT | PTE_RW | PTE_ACCESSED;
                    pt_bytes[pte_off..pte_off + 4]
                        .copy_from_slice(&pte.to_le_bytes());

                    va += page_size as u64;
                    pa += page_size as u64;
                }
            }

            layout.set_pt_size(pt_bytes.len())?;
            memory.extend(&pt_bytes);
        };

        let exn_stack_top_gva = hyperlight_common::layout::MAX_GVA as u64
            - hyperlight_common::layout::SCRATCH_TOP_EXN_STACK_OFFSET
            + 1;

        let extra_regions = Vec::new();
        let hash = hash(&memory, &extra_regions)?;

        let sregs = None;

        Ok(Self {
            sandbox_id: SANDBOX_CONFIGURATION_COUNTER.fetch_add(1, Ordering::Relaxed),
            memory: ReadonlySharedMemory::from_bytes(&memory)?,
            layout,
            regions: extra_regions,
            load_info,
            hash,
            stack_top_gva: exn_stack_top_gva,
            sregs,
            separate_pt_bytes: Vec::new(),
            entrypoint: NextAction::Initialise(load_addr + entrypoint_va - base_va),
        })
    }

    // It might be nice to consider moving at least stack_top_gva into
    // layout, and sharing (via RwLock or similar) the layout between
    // the (host-side) mem mgr (where it can be passed in here) and
    // the sandbox vm itself (which modifies it as it receives
    // requests from the sandbox).
    #[allow(clippy::too_many_arguments)]
    /// Take a snapshot of the memory in `shared_mem`, then create a new
    /// instance of `Self` with the snapshot stored therein.
    #[instrument(err(Debug), skip_all, parent = Span::current(), level= "Trace")]
    pub(crate) fn new<S: SharedMemory>(
        shared_mem: &mut SnapshotSharedMemory<S>,
        scratch_mem: &mut S,
        sandbox_id: u64,
        mut layout: SandboxMemoryLayout,
        load_info: LoadInfo,
        regions: Vec<MemoryRegion>,
        root_pt_gpas: &[u64],
        stack_top_gva: u64,
        sregs: CommonSpecialRegisters,
        entrypoint: NextAction,
    ) -> Result<Self> {
        use std::collections::HashMap;
        let mut phys_seen = HashMap::<u64, usize>::new();
        let memory = shared_mem.with_contents(|snap_c| {
            scratch_mem.with_contents(|scratch_c| {
                // Build CoW resolution map (i686 only): maps original GPAs
                // to their CoW'd scratch GPAs so the PT walker can read the
                // actual page table data instead of stale snapshot copies.
                #[cfg(feature = "i686-guest")]
                let cow_map = {
                    let kernel_root = root_pt_gpas.first().copied().unwrap_or(0);
                    build_cow_map(snap_c, scratch_c, layout, kernel_root)
                };

                // Pass 1: count how many pages need to live
                let live_pages = filtered_mappings(
                    snap_c,
                    scratch_c,
                    &regions,
                    layout,
                    root_pt_gpas,
                    #[cfg(feature = "i686-guest")]
                    &cow_map,
                );

                // Pass 2: copy live pages and build new page tables
                let mut snapshot_memory: Vec<u8> = Vec::new();
                #[allow(unused_mut)]
                let mut pt_bytes: Vec<u8> = Vec::new();

                #[cfg(not(feature = "i686-guest"))]
                {
                    let pt_buf = GuestPageTableBuffer::new(layout.get_pt_base_gpa() as usize);
                    for (mapping, contents) in live_pages {
                        let kind = match mapping.kind {
                            MappingKind::Cow(cm) => MappingKind::Cow(cm),
                            MappingKind::Basic(bm) if bm.writable => {
                                MappingKind::Cow(CowMapping {
                                    readable: bm.readable,
                                    executable: bm.executable,
                                })
                            }
                            MappingKind::Basic(bm) => MappingKind::Basic(BasicMapping {
                                readable: bm.readable,
                                writable: false,
                                executable: bm.executable,
                            }),
                            MappingKind::Unmapped => continue,
                        };
                        let new_gpa =
                            phys_seen.entry(mapping.phys_base).or_insert_with(|| {
                                let new_offset = snapshot_memory.len();
                                snapshot_memory.extend(contents);
                                new_offset + SandboxMemoryLayout::BASE_ADDRESS
                            });
                        let mapping = Mapping {
                            phys_base: *new_gpa as u64,
                            virt_base: mapping.virt_base,
                            len: PAGE_SIZE as u64,
                            kind,
                        };
                        unsafe { vmem::map(&pt_buf, mapping) };
                    }
                    // Phase 3: Map the special mappings
                    map_specials(&pt_buf, layout.get_scratch_size());
                    let pt_data = pt_buf.into_bytes();
                    layout.set_pt_size(pt_data.len())?;
                    snapshot_memory.extend(&pt_data);
                }

                #[cfg(feature = "i686-guest")]
                {
                    // Compacting snapshot for i686: live pages packed densely
                    // starting at BASE_ADDRESS. Page tables rebuilt with
                    // compacted GPAs + CoW bits.
                    use hyperlight_common::i686_guest_vmem::{
                        PAGE_ACCESSED, PAGE_AVL_COW, PAGE_PRESENT, PAGE_RW, PAGE_USER,
                    };
                    let page_size: usize = 4096;

                    // Phase 1: pack live pages densely (same as x86_64).
                    for (mapping, contents) in live_pages {
                        if matches!(mapping.kind, MappingKind::Unmapped) {
                            continue;
                        }
                        phys_seen.entry(mapping.phys_base).or_insert_with(|| {
                            let new_offset = snapshot_memory.len();
                            snapshot_memory.extend(contents);
                            new_offset + SandboxMemoryLayout::BASE_ADDRESS
                        });
                    }

                    // Phase 2: build per-process i686 PTs with compacted GPAs.
                    let pd_base_gpa = layout.get_pt_base_gpa() as usize;
                    let n_roots = root_pt_gpas.len().max(1);
                    pt_bytes.resize(n_roots * page_size, 0); // PD pages

                    let scratch_size = layout.get_scratch_size();
                    let scratch_gpa =
                        hyperlight_common::layout::scratch_base_gpa(scratch_size);

                    let read_u32 = |gpa: u64| -> u32 {
                        // Resolve CoW'd GPAs through the kernel PD.
                        let resolved_gpa = {
                            let page_gpa = gpa & 0xFFFFF000;
                            if let Some(&scratch_gpa_val) = cow_map.get(&page_gpa) {
                                scratch_gpa_val + (gpa & 0xFFF)
                            } else {
                                gpa
                            }
                        };
                        access_gpa(snap_c, scratch_c, layout, resolved_gpa)
                            .and_then(|(mem, off)| mem.get(off..off + 4))
                            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .unwrap_or(0)
                    };

                    // Helper: rebuild a PT page with compacted frame GPAs.
                    let rebuild_pt =
                        |pt_bytes: &mut Vec<u8>,
                         old_pt_gpa: u64,
                         user_bit: u32,
                         phys_map: &HashMap<u64, usize>| -> u32 {
                            let new_pt_offset = pt_bytes.len();
                            pt_bytes.resize(new_pt_offset + page_size, 0);
                            let new_pt_gpa = (pd_base_gpa + new_pt_offset) as u32;
                            for pti in 0..1024usize {
                                let pte = read_u32(old_pt_gpa + pti as u64 * 4);
                                if (pte & (PAGE_PRESENT as u32)) == 0 {
                                    continue;
                                }
                                let old_frame = (pte & 0xFFFFF000) as u64;
                                // All pages (including those CoW'd to scratch during
                                // first boot) get compacted into the snapshot region.
                                let new_frame =
                                    if let Some(&new_gpa) = phys_map.get(&old_frame) {
                                        new_gpa as u32
                                    } else {
                                        continue;
                                    };
                                let mut flags = (pte & 0xFFF) | user_bit;
                                // Mark writable pages with CoW bit for post-restore.
                                if (flags & (PAGE_RW as u32)) != 0
                                    || (flags & (PAGE_AVL_COW as u32)) != 0
                                {
                                    flags = (flags & !(PAGE_RW as u32))
                                        | (PAGE_AVL_COW as u32);
                                }
                                let off = new_pt_offset + pti * 4;
                                pt_bytes[off..off + 4]
                                    .copy_from_slice(&(new_frame | flags).to_le_bytes());
                            }
                            new_pt_gpa
                        };

                    // Kernel PTs from first root.
                    let first_root = root_pt_gpas.first().copied().unwrap_or(0);
                    let mut kernel_pdes = [(0u32, 0u32); 256];
                    for pdi in 0..256usize {
                        let pde = read_u32(first_root + pdi as u64 * 4);
                        if (pde & (PAGE_PRESENT as u32)) == 0 {
                            continue;
                        }
                        let pt_gpa = (pde & 0xFFFFF000) as u64;
                        let new_pt_gpa =
                            rebuild_pt(&mut pt_bytes, pt_gpa, 0, &phys_seen);
                        let fixed_pde = (pde & 0xFFF) | new_pt_gpa;
                        kernel_pdes[pdi] = (fixed_pde, new_pt_gpa);
                    }

                    // Resolve a VA through a PD to find the CoW-resolved PA.
                    // Used to find where CoW'd page table pages actually live.
                    let resolve_va_through_pd = |pd_gpa: u64, va: u64| -> u64 {
                        let pdi = ((va >> 22) & 0x3FF) as u64;
                        let pde = read_u32(pd_gpa + pdi * 4);
                        if (pde & (PAGE_PRESENT as u32)) == 0 {
                            return va; // not mapped, return as-is
                        }
                        let pt_gpa = (pde & 0xFFFFF000) as u64;
                        let pti = ((va >> 12) & 0x3FF) as u64;
                        let pte = read_u32(pt_gpa + pti * 4);
                        if (pte & (PAGE_PRESENT as u32)) == 0 {
                            return va;
                        }
                        (pte & 0xFFFFF000) as u64
                    };

                    // Per-process PDs.
                    for (root_idx, &root) in root_pt_gpas.iter().enumerate() {
                        let pd_offset = root_idx * page_size;
                        for pdi in 0..256usize {
                            let (fixed_pde, _) = kernel_pdes[pdi];
                            if fixed_pde != 0 {
                                pt_bytes[pd_offset + pdi * 4..pd_offset + pdi * 4 + 4]
                                    .copy_from_slice(&fixed_pde.to_le_bytes());
                            }
                        }
                        for pdi in 256..1024usize {
                            let pde = read_u32(root + pdi as u64 * 4);
                            if (pde & (PAGE_PRESENT as u32)) == 0 {
                                continue;
                            }
                            let user_bit = PAGE_USER as u32;
                            let pt_gpa_raw = (pde & 0xFFFFF000) as u64;
                            // The PDE may contain a pre-CoW address (e.g., kpool).
                            // Resolve through the kernel PD to find the actual PT location.
                            let pt_gpa = resolve_va_through_pd(first_root, pt_gpa_raw);
                            let new_pt_gpa =
                                rebuild_pt(&mut pt_bytes, pt_gpa, user_bit, &phys_seen);
                            let fixed_pde = (pde & 0xFFF) | new_pt_gpa | user_bit;
                            pt_bytes[pd_offset + pdi * 4..pd_offset + pdi * 4 + 4]
                                .copy_from_slice(&fixed_pde.to_le_bytes());
                        }
                    }

                    // Map scratch region into every PD (writable, identity).
                    {
                        let sf = (PAGE_PRESENT | PAGE_RW | PAGE_ACCESSED) as u32;
                        let scratch_pages = scratch_size / page_size;
                        for ri in 0..n_roots {
                            let pd_off = ri * page_size;
                            for pi in 0..scratch_pages {
                                let gpa = scratch_gpa + (pi * page_size) as u64;
                                let pdi = ((gpa >> 22) & 0x3FF) as usize;
                                let pti = ((gpa >> 12) & 0x3FF) as usize;
                                let pde_off = pd_off + pdi * 4;
                                let epde = u32::from_le_bytes([
                                    pt_bytes[pde_off], pt_bytes[pde_off+1],
                                    pt_bytes[pde_off+2], pt_bytes[pde_off+3],
                                ]);
                                let pt_off = if (epde & (PAGE_PRESENT as u32)) == 0 {
                                    let o = pt_bytes.len();
                                    pt_bytes.resize(o + page_size, 0);
                                    let g = (pd_base_gpa + o) as u32;
                                    pt_bytes[pde_off..pde_off+4]
                                        .copy_from_slice(&(g | sf).to_le_bytes());
                                    o
                                } else {
                                    ((epde & 0xFFFFF000) as usize) - pd_base_gpa
                                };
                                let pte_off = pt_off + pti * 4;
                                pt_bytes[pte_off..pte_off+4]
                                    .copy_from_slice(&((gpa as u32) | sf).to_le_bytes());
                            }
                        }
                    }

                    // Identity-map the entire compacted snapshot region into
                    // every PD (writable). This ensures the kernel's software
                    // page-walker can dereference compacted user frame addresses
                    // (which may extend beyond the original MEMORY_SIZE) when
                    // walking page tables.
                    {
                        let sf = (PAGE_PRESENT | PAGE_RW | PAGE_ACCESSED) as u32;
                        let snapshot_end = SandboxMemoryLayout::BASE_ADDRESS + snapshot_memory.len();
                        let snapshot_pages =
                            (snapshot_end - SandboxMemoryLayout::BASE_ADDRESS) / page_size;
                        for ri in 0..n_roots {
                            let pd_off = ri * page_size;
                            for pi in 0..snapshot_pages {
                                let gpa = (SandboxMemoryLayout::BASE_ADDRESS + pi * page_size) as u64;
                                let pdi = ((gpa >> 22) & 0x3FF) as usize;
                                let pti = ((gpa >> 12) & 0x3FF) as usize;
                                let pde_off = pd_off + pdi * 4;
                                let epde = u32::from_le_bytes([
                                    pt_bytes[pde_off], pt_bytes[pde_off+1],
                                    pt_bytes[pde_off+2], pt_bytes[pde_off+3],
                                ]);
                                let pt_off = if (epde & (PAGE_PRESENT as u32)) == 0 {
                                    let o = pt_bytes.len();
                                    pt_bytes.resize(o + page_size, 0);
                                    let g = (pd_base_gpa + o) as u32;
                                    pt_bytes[pde_off..pde_off+4]
                                        .copy_from_slice(&(g | sf).to_le_bytes());
                                    o
                                } else {
                                    ((epde & 0xFFFFF000) as usize) - pd_base_gpa
                                };
                                let pte_off = pt_off + pti * 4;
                                // Only set if PTE is currently empty — don't
                                // overwrite legitimate user/kernel mappings.
                                let cur = u32::from_le_bytes([
                                    pt_bytes[pte_off], pt_bytes[pte_off+1],
                                    pt_bytes[pte_off+2], pt_bytes[pte_off+3],
                                ]);
                                if (cur & (PAGE_PRESENT as u32)) == 0 {
                                    pt_bytes[pte_off..pte_off+4]
                                        .copy_from_slice(&((gpa as u32) | sf).to_le_bytes());
                                }
                            }
                        }
                    }

                    layout.set_pt_size(pt_bytes.len())?;
                    // Store PT bytes separately to avoid growing the blob
                    // past the original shared_mem size (which would overlap
                    // with map_file_cow regions placed right after it).
                }

                Ok::<(Vec<u8>, Vec<u8>), crate::HyperlightError>((snapshot_memory, pt_bytes))
            })
        })???;
        let (memory, separate_pt_bytes) = memory;
        layout.set_snapshot_size(memory.len());

        // For non-compacting i686 snapshots, keep the regions so the RAMFS
        // and other map_file_cow mappings are accessible after restore.
        #[cfg(feature = "i686-guest")]
        let regions = regions;
        #[cfg(not(feature = "i686-guest"))]
        let regions = Vec::new();

        let hash = hash(&memory, &regions)?;
        Ok(Self {
            sandbox_id,
            layout,
            memory: ReadonlySharedMemory::from_bytes(&memory)?,
            regions,
            load_info,
            hash,
            stack_top_gva,
            sregs: Some(sregs),
            separate_pt_bytes,
            entrypoint,
        })
    }

    /// The id of the sandbox this snapshot was taken from.
    pub(crate) fn sandbox_id(&self) -> u64 {
        self.sandbox_id
    }

    /// Get the mapped regions from this snapshot
    pub(crate) fn regions(&self) -> &[MemoryRegion] {
        &self.regions
    }

    /// Return the main memory contents of the snapshot
    #[instrument(skip_all, parent = Span::current(), level= "Trace")]
    pub(crate) fn memory(&self) -> &ReadonlySharedMemory {
        &self.memory
    }

    /// Return a copy of the load info for the exe in the snapshot
    pub(crate) fn load_info(&self) -> LoadInfo {
        self.load_info.clone()
    }

    pub(crate) fn layout(&self) -> &crate::mem::layout::SandboxMemoryLayout {
        &self.layout
    }

    pub(crate) fn root_pt_gpa(&self) -> u64 {
        // Page tables are rebuilt in scratch at the layout's PT base during
        // snapshot compaction. On restore, update_scratch_bookkeeping copies
        // them back to scratch. CR3 should point to this address.
        self.layout.get_pt_base_gpa()
    }

    pub(crate) fn stack_top_gva(&self) -> u64 {
        self.stack_top_gva
    }

    /// Returns the special registers stored in this snapshot.
    /// Returns None for snapshots created directly from a binary (before preinitialisation).
    /// Returns Some for snapshots taken from a running sandbox.
    /// Note: The CR3 value in the returned struct should NOT be used for restore;
    /// use `root_pt_gpa()` instead since page tables are relocated during snapshot.
    pub(crate) fn sregs(&self) -> Option<&CommonSpecialRegisters> {
        self.sregs.as_ref()
    }

    pub(crate) fn separate_pt_bytes(&self) -> &[u8] {
        &self.separate_pt_bytes
    }

    pub(crate) fn entrypoint(&self) -> NextAction {
        self.entrypoint
    }
}

impl PartialEq for Snapshot {
    fn eq(&self, other: &Snapshot) -> bool {
        self.hash == other.hash
    }
}

#[cfg(test)]
mod tests {
    use hyperlight_common::vmem::{self, BasicMapping, Mapping, MappingKind, PAGE_SIZE};

    use crate::hypervisor::regs::CommonSpecialRegisters;
    use crate::mem::exe::LoadInfo;
    use crate::mem::layout::SandboxMemoryLayout;
    use crate::mem::mgr::{GuestPageTableBuffer, SandboxMemoryManager, SnapshotSharedMemory};
    use crate::mem::shared_mem::{
        ExclusiveSharedMemory, HostSharedMemory, ReadonlySharedMemory, SharedMemory,
    };

    fn default_sregs() -> CommonSpecialRegisters {
        CommonSpecialRegisters::default()
    }

    const SIMPLE_PT_BASE: usize = PAGE_SIZE + SandboxMemoryLayout::BASE_ADDRESS;

    fn make_simple_pt_mem(contents: &[u8]) -> SnapshotSharedMemory<ExclusiveSharedMemory> {
        let pt_buf = GuestPageTableBuffer::new(SIMPLE_PT_BASE);
        let mapping = Mapping {
            phys_base: SandboxMemoryLayout::BASE_ADDRESS as u64,
            virt_base: SandboxMemoryLayout::BASE_ADDRESS as u64,
            len: PAGE_SIZE as u64,
            kind: MappingKind::Basic(BasicMapping {
                readable: true,
                writable: true,
                executable: true,
            }),
        };
        unsafe { vmem::map(&pt_buf, mapping) };
        super::map_specials(&pt_buf, PAGE_SIZE);
        let pt_bytes = pt_buf.into_bytes();

        let mut snapshot_mem = vec![0u8; PAGE_SIZE + pt_bytes.len()];
        snapshot_mem[0..PAGE_SIZE].copy_from_slice(contents);
        snapshot_mem[PAGE_SIZE..].copy_from_slice(&pt_bytes);
        ReadonlySharedMemory::from_bytes(&snapshot_mem)
            .unwrap()
            .to_mgr_snapshot_mem()
            .unwrap()
    }

    fn make_simple_pt_mgr() -> (SandboxMemoryManager<HostSharedMemory>, u64) {
        let cfg = crate::sandbox::SandboxConfiguration::default();
        let scratch_mem = ExclusiveSharedMemory::new(cfg.get_scratch_size()).unwrap();
        let mgr = SandboxMemoryManager::new(
            SandboxMemoryLayout::new(cfg, 4096, 0x3000, None).unwrap(),
            make_simple_pt_mem(&[0u8; PAGE_SIZE]),
            scratch_mem,
            super::NextAction::None,
        );
        let (mgr, _) = mgr.build().unwrap();
        (mgr, SIMPLE_PT_BASE as u64)
    }

    #[test]
    fn multiple_snapshots_independent() {
        let (mut mgr, pt_base) = make_simple_pt_mgr();

        // Create first snapshot with pattern A
        let pattern_a = vec![0xAA; PAGE_SIZE];
        let snapshot_a = super::Snapshot::new(
            &mut make_simple_pt_mem(&pattern_a).build().0,
            &mut mgr.scratch_mem,
            1,
            mgr.layout,
            LoadInfo::dummy(),
            Vec::new(),
            pt_base,
            0,
            default_sregs(),
            super::NextAction::None,
        )
        .unwrap();

        // Create second snapshot with pattern B
        let pattern_b = vec![0xBB; PAGE_SIZE];
        let snapshot_b = super::Snapshot::new(
            &mut make_simple_pt_mem(&pattern_b).build().0,
            &mut mgr.scratch_mem,
            2,
            mgr.layout,
            LoadInfo::dummy(),
            Vec::new(),
            pt_base,
            0,
            default_sregs(),
            super::NextAction::None,
        )
        .unwrap();

        // Restore snapshot A
        mgr.restore_snapshot(&snapshot_a).unwrap();
        mgr.shared_mem
            .with_contents(|contents| assert_eq!(&contents[0..pattern_a.len()], &pattern_a[..]))
            .unwrap();

        // Restore snapshot B
        mgr.restore_snapshot(&snapshot_b).unwrap();
        mgr.shared_mem
            .with_contents(|contents| assert_eq!(&contents[0..pattern_b.len()], &pattern_b[..]))
            .unwrap();
    }
}
