use tarnik::wasm;
use tarnik_ast::{Signature, WasmType, WatModule};

fn main() {
    let mut module: WatModule = wasm! {
        fn run() {
            let x: i32 = 0;
        }
    };

    module.add_memory("$memory", 1);
    module.add_export("memory", "memory", "$memory");
    module.add_export("_start", "func", "$run");
    module.add_import(
        "console",
        "log",
        WasmType::func(vec![WasmType::I32, WasmType::I32], None),
    );

    println!("{module}");
}
