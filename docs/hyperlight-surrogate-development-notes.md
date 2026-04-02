### HyperlightSurrogate

`hyperlight_surrogate.exe` is a tiny Rust application we use to create multiple virtual machine (VM) partitions per process when running on Windows with the Windows Hypervisor Platform (WHP, e-g Hyper-V). This binary has no functionality. Its purpose is to provide a running process into which memory will be mapped via the `WHvMapGpaRange2` Windows API. Hyperlight does this memory mapping to pass parameters into, and fetch return values out of, a given VM partition.

> Note: The use of surrogates is a temporary workaround on Windows until WHP allows us to create more than one partition per running process.

These surrogate processes are managed by the host via the [surrogate_process_manager](./src/hyperlight_host/src/hypervisor/surrogate_process_manager.rs) which will launch several of these surrogates (up to the 512), assign memory to them, then launch partitions from there, and reuse them as necessary.

`hyperlight_surrogate.exe` gets built during `hyperlight-host`'s build script, gets embedded into the `hyperlight-host` Rust library via [rust-embed](https://crates.io/crates/rust-embed), and is extracted at runtime next to the executable when the surrogate process manager is initialized.