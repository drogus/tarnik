use tarnik::wasm;
use tarnik_ast::{Signature, WasmType, WatModule};

fn main() {
    let mut module: WatModule = wasm! {
         type String = [mut i8];

        fn run() {
            let str: String;
            str = "foo";
            str[2] = 'i';
            if 1 == 1 {

            } else {

            }
            assert(0, "This should throw");
        }
    };

    module.add_memory("$memory", 1);
    module.add_export("memory", "memory", "$memory");
    module.add_export("_start", "func", "$run");
    // (import "env" "assert_exception_tag" (tag $AssertException (param i32 i32)));
    module.add_import(
        "env",
        "assert_exception_tag",
        WasmType::Tag {
            name: "$AssertException".to_string(),
            signature: Box::new(Signature {
                params: vec![WasmType::I32, WasmType::I32],
                result: None,
            }),
        },
    );

    println!("{module}");
}
