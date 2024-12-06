fn main() {
    let module: tarnik_ast::WatModule = tarnik::wasm! {
        #[export("memory")]
        memory!("memory", 1);

        #[import("wasi_snapshot_preview1", "fd_write")]
        fn write(fd: i32, iov_start: i32, iov_len: i32, nwritten: i32) -> i32;

        #[import("wasi_snapshot_preview1", "fd_read")]
        fn read(fd: i32, iov_start: i32, iov_len: i32, nread: i32) -> i32;

        type I64Array = [i64];
        fn run() {
            let x: I64Array = [44];
            test_any(x);
        }

        fn test_any(numbers: anyref) {
            if ref_test!(numbers, I64Array) {
                let numbers_i64: I64Array = numbers as I64Array;
                assert(numbers_i64[0] == 44 as i64, "The first element of the numbers array should be 44");
            } else {
                assert(0, "$numbers should be an I64Array");
            }
        }
    };

    println!("{module}");
}
