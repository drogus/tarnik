use tarnik::wasm;
use tarnik_ast::{Signature, WasmType, WatModule};

fn main() {
    let mut module: WatModule = wasm! {
        #[export("memory")]
        memory!("memory", 1);

        #[export("_start")]
        fn run() {
            let x: i32 = 0;
            x = -1;
        }
    };

    // module.add_export("_start", "func", "$run");
    module.add_import(
        "console",
        "log",
        WasmType::func(vec![WasmType::I32, WasmType::I32], None),
    );

    println!("{module}");
}
