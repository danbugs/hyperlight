#![no_std]
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

extern crate alloc;

// TODO(danbugs): this is needed so the panic handler is actually brought in.
// We can remove it later once more functionality was moved to hyperlight-guest-bin
extern crate hyperlight_guest_bin;

pub mod dispatch;
pub mod error;
pub mod flatbuffer;
pub mod logging;
pub mod types;
