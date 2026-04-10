/*
Copyright 2026  The Hyperlight Authors.

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

//! i686 guest page-table definitions used by the x86_64 host.
//!
//! `arch/i686/vmem.rs` is only compiled when `target_arch = "x86"` (guest
//! side), so the host cannot reach its definitions through `mod arch`. This
//! module provides the canonical i686 PTE flag constants and the host-side
//! walker used by snapshot compaction.

use crate::vmem::{BasicMapping, CowMapping, Mapping, MappingKind, TableReadOps};

// i686 PTE flags (u64 to match the TableReadOps trait interface).
pub const PAGE_PRESENT: u64 = 1;
pub const PAGE_RW: u64 = 1 << 1;
pub const PAGE_USER: u64 = 1 << 2;
pub const PAGE_ACCESSED: u64 = 1 << 5;
pub const PAGE_AVL_COW: u64 = 1 << 9;
pub const PTE_ADDR_MASK: u64 = 0xFFFFF000;

pub const PAGE_SIZE: usize = 4096;

/// Walk an i686 2-level page table and return all present mappings.
///
/// # Safety
/// The caller must ensure that `op` provides valid page table memory.
pub unsafe fn virt_to_phys_all<Op: TableReadOps>(op: &Op) -> alloc::vec::Vec<Mapping> {
    let root = op.root_table();
    let mut mappings = alloc::vec::Vec::new();
    for pdi in 0..1024u64 {
        let pde_ptr = Op::entry_addr(root, pdi * 4);
        let pde: u64 = unsafe { op.read_entry(pde_ptr) } as u64;
        if (pde & PAGE_PRESENT) == 0 {
            continue;
        }
        let pt_phys = pde & PTE_ADDR_MASK;
        let pt_base = Op::from_phys(pt_phys as crate::vmem::PhysAddr);
        for pti in 0..1024u64 {
            let pte_ptr = Op::entry_addr(pt_base, pti * 4);
            let pte: u64 = unsafe { op.read_entry(pte_ptr) } as u64;
            if (pte & PAGE_PRESENT) == 0 {
                continue;
            }
            let phys_base = pte & PTE_ADDR_MASK;
            let virt_base = (pdi << 22) | (pti << 12);
            let kind = if (pte & PAGE_AVL_COW) != 0 {
                MappingKind::Cow(CowMapping {
                    readable: true,
                    executable: true,
                })
            } else {
                MappingKind::Basic(BasicMapping {
                    readable: true,
                    writable: (pte & PAGE_RW) != 0,
                    executable: true,
                })
            };
            mappings.push(Mapping {
                phys_base,
                virt_base,
                len: PAGE_SIZE as u64,
                kind,
            });
        }
    }
    mappings
}
