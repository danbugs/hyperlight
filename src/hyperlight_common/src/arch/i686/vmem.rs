/*
Copyright 2025  The Hyperlight Authors.

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

//! i686 2-level page table manipulation code.
//!
//! This module implements page table setup for i686 protected mode using 2-level paging:
//! - PD (Page Directory) - bits 31:22 - 1024 entries × 4 bytes, each covering 4MB
//! - PT (Page Table) - bits 21:12 - 1024 entries × 4 bytes, each covering 4KB pages

use crate::vmem::{
    BasicMapping, CowMapping, Mapping, MappingKind, TableOps, TableReadOps, Void,
};

pub const PAGE_SIZE: usize = 4096;
pub const PAGE_TABLE_SIZE: usize = 4096;
pub type PageTableEntry = u32;
pub type VirtAddr = u32;
pub type PhysAddr = u32;

// i686 PTE flags (stored in low 12 bits of the 32-bit PTE, but we use u64
// for the trait interface and truncate on read/write)
const PAGE_PRESENT: u64 = 1;
const PAGE_RW: u64 = 1 << 1;
const PAGE_ACCESSED: u64 = 1 << 5;
const PAGE_AVL_COW: u64 = 1 << 9;
const PTE_ADDR_MASK: u64 = 0xFFFFF000;

/// Map a region into the 2-level page table.
///
/// # Safety
/// The caller must ensure that `op` provides valid page table memory.
pub unsafe fn map<Op: TableOps>(op: &Op, mapping: Mapping) {
    if matches!(mapping.kind, MappingKind::Unmapped) {
        return;
    }

    let flags = match &mapping.kind {
        MappingKind::Basic(BasicMapping { writable, .. }) => {
            PAGE_PRESENT | PAGE_ACCESSED | if *writable { PAGE_RW } else { 0 }
        }
        MappingKind::Cow(CowMapping { .. }) => {
            PAGE_PRESENT | PAGE_ACCESSED | PAGE_AVL_COW
        }
        MappingKind::Unmapped => 0,
    };

    let mut va = mapping.virt_base;
    let mut pa = mapping.phys_base;
    let end = mapping.virt_base + mapping.len;

    while va < end {
        let pdi = ((va >> 22) & 0x3FF) as u64;
        let pti = ((va >> 12) & 0x3FF) as u64;

        // Read PDE from the root (page directory)
        let root = op.root_table();
        let pde_ptr = Op::entry_addr(root, pdi * 4); // 4 bytes per entry
        let pde: u64 = unsafe { op.read_entry(pde_ptr) } as u64;

        // If PDE not present, allocate a new page table
        let pt_phys: u64 = if (pde & PAGE_PRESENT) == 0 {
            let new_pt = unsafe { op.alloc_table() };
            let new_pt_phys: u64 = Op::to_phys(new_pt) as u64;
            let new_pde: PageTableEntry =
                (new_pt_phys | PAGE_PRESENT | PAGE_RW | PAGE_ACCESSED) as PageTableEntry;
            let _ = unsafe { op.write_entry(pde_ptr, new_pde) };
            new_pt_phys
        } else {
            pde & PTE_ADDR_MASK
        };

        // Write PT entry
        let pt_base = Op::from_phys(pt_phys as PhysAddr);
        let pte_ptr = Op::entry_addr(pt_base, pti * 4); // 4 bytes per entry
        let pte: PageTableEntry = (pa | flags) as PageTableEntry;
        let _ = unsafe { op.write_entry(pte_ptr, pte) };

        va += PAGE_SIZE as u64;
        pa += PAGE_SIZE as u64;
    }
}

/// Walk the page table and yield mappings for the given virtual address.
///
/// # Safety
/// The caller must ensure that `op` provides valid page table memory.
pub unsafe fn virt_to_phys<Op: TableReadOps>(
    op: &Op,
    address: u64,
) -> impl Iterator<Item = Mapping> {
    let pdi = ((address >> 22) & 0x3FF) as u64;
    let pti = ((address >> 12) & 0x3FF) as u64;

    let root = op.root_table();
    let pde_ptr = Op::entry_addr(root, pdi * 4);
    let pde: u64 = unsafe { op.read_entry(pde_ptr) } as u64;

    if (pde & PAGE_PRESENT) == 0 {
        return core::iter::once(Mapping {
            phys_base: 0,
            virt_base: address & !0xFFF,
            len: PAGE_SIZE as u64,
            kind: MappingKind::Unmapped,
        });
    }

    let pt_phys = (pde & PTE_ADDR_MASK) as PhysAddr;
    let pt_base = Op::from_phys(pt_phys);
    let pte_ptr = Op::entry_addr(pt_base, pti * 4);
    let pte: u64 = unsafe { op.read_entry(pte_ptr) } as u64;

    if (pte & PAGE_PRESENT) == 0 {
        return core::iter::once(Mapping {
            phys_base: 0,
            virt_base: address & !0xFFF,
            len: PAGE_SIZE as u64,
            kind: MappingKind::Unmapped,
        });
    }

    let phys_base = pte & PTE_ADDR_MASK;
    let kind = if (pte & PAGE_AVL_COW) != 0 {
        MappingKind::Cow(CowMapping {
            readable: true,
            executable: true, // No NX on i686
        })
    } else {
        MappingKind::Basic(BasicMapping {
            readable: true,
            writable: (pte & PAGE_RW) != 0,
            executable: true, // No NX on i686
        })
    };

    core::iter::once(Mapping {
        phys_base,
        virt_base: address & !0xFFF,
        len: PAGE_SIZE as u64,
        kind,
    })
}

pub trait TableMovability<Op: TableReadOps + ?Sized, TableMoveInfo> {}
impl<Op: TableOps<TableMovability = crate::vmem::MayMoveTable>> TableMovability<Op, Op::TableAddr>
    for crate::vmem::MayMoveTable
{
}
impl<Op: TableReadOps> TableMovability<Op, Void> for crate::vmem::MayNotMoveTable {}
