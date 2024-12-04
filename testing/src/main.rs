use tarnik::wasm;
use tarnik_ast::{Signature, WasmType, WatModule};

fn main() {
    let mut module: WatModule = wasm! {
        #[export("memory")]
        memory!("memory", 1);

        type Foo = [i32];

        #[export("_start")]
        fn run() {
            let x: Nullable<Foo> = [0, 1, 2];
            let y: Foo = x as Foo;
            let a: i32 = 0;
            let b: i64 = a as i64;
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
