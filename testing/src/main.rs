use tarnik_ast::{Signature, WasmType};

fn main() {
    let mut module: tarnik_ast::WatModule = tarnik::wasm! {
        #[export("memory")]
        memory!("memory", 1);

        #[import("wasi_snapshot_preview1", "fd_write")]
        fn write(fd: i32, iov_start: i32, iov_len: i32, nwritten: i32) -> i32;

        #[import("wasi_snapshot_preview1", "fd_read")]
        fn read(fd: i32, iov_start: i32, iov_len: i32, nread: i32) -> i32;

        type FooFunc = fn(i32) -> i32;

        type JSExceptionType = fn(anyref);
        tag!(JSException, JSExceptionType);

        #[export("_start")]
        fn run() {
            if 1 {
                if 2 {
                    try {
                        try {

                        }
                        catch(JSException, error: anyref) {

                        }
                    }
                    catch(JSException, error: anyref) {

                    }
                }
            }
        }
    };
    module.add_import(
        "env",
        "assert_exception_tag",
        WasmType::Tag {
            name: "$AssertException".to_string(),
            signature: Box::new(Signature {
                params: vec![(None, WasmType::I32), (None, WasmType::I32)],
                result: None,
            }),
        },
    );

    println!("{module}");
}
