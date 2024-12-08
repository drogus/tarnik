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
            let offset1: i32 = data!("foo");
            let offset2: i32 = data!("bar");
            let offset3: i32 = data!("foo");

            assert(offset1 == offset3, "data entries should not add a new entry if one exists");
            assert(memory[offset2] == 'b', "offset should point at the data");
            assert(memory[offset2 + 1] == 'a', "offset should point at the data");
            assert(memory[offset2 + 2] == 'r', "offset should point at the data");
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
