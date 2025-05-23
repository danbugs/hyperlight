/*
Copyright 2024 The Hyperlight Authors.

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

#![no_std]

extern crate alloc;

use alloc::string::ToString;

use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;
use hyperlight_guest::entrypoint::abort_with_code_and_message;

// It looks like rust-analyzer doesn't correctly manage no_std crates,
// and so it displays an error about a duplicate panic_handler.
// See more here: https://github.com/rust-lang/rust-analyzer/issues/4490
// The cfg_attr attribute is used to avoid clippy failures as test pulls in std which pulls in a panic handler
#[cfg_attr(not(test), panic_handler)]
#[allow(clippy::panic)]
// to satisfy the clippy when cfg == test
#[allow(dead_code)]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let msg = info.to_string();
    let c_string = alloc::ffi::CString::new(msg)
        .unwrap_or_else(|_| alloc::ffi::CString::new("panic (invalid utf8)").unwrap());

    unsafe { abort_with_code_and_message(&[ErrorCode::UnknownError as u8], c_string.as_ptr()) }
}
