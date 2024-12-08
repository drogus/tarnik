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

        #[export("_start")]
        fn run() {
            let foo_ref: FooFunc = foo;

            let y: i32 = foo_ref(33);

            assert(y == 44, "it should be possible to call a func ref");
        }

        fn foo(x: i32) -> i32 {
            return x + 11;
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
