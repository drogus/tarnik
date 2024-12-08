use tarnik_ast::{Signature, WasmType};

fn main() {
    let mut module: tarnik_ast::WatModule = tarnik::wasm! {
        #[export("memory")]
        memory!("memory", 1);

        #[import("wasi_snapshot_preview1", "fd_write")]
        fn write(fd: i32, iov_start: i32, iov_len: i32, nwritten: i32) -> i32;

        #[import("wasi_snapshot_preview1", "fd_read")]
        fn read(fd: i32, iov_start: i32, iov_len: i32, nread: i32) -> i32;

        rec! {
            struct HashMapEntry {
                key: i32,
                value: anyref
            }
            type EntriesArray = [mut Nullable<HashMapEntry>];
            struct HashMap {
                entries: mut EntriesArray,
                size: mut i32
            }
        }

        static mut foo: HashMap = new_hashmap();

        fn new_hashmap() -> HashMap {
            return HashMap { entries: [null; 10], size: 0 };
        }

        #[export("_start")]
        fn run() {
        }
    };
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
