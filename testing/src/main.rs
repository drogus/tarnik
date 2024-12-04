fn main() {
    let module: tarnik_ast::WatModule = tarnik::wasm! {
        #[export("memory")]
        memory!("memory", 1);

        #[import("wasi_snapshot_preview1", "fd_write")]
        fn write(fd: i32, iov_start: i32, iov_len: i32, nwritten: i32) -> i32;

        #[import("wasi_snapshot_preview1", "fd_read")]
        fn read(fd: i32, iov_start: i32, iov_len: i32, nread: i32) -> i32;

        type ImmutableString = [i8];

        #[export("_start")]
        fn run() {
            // we will read at most 100 chars into memory offset 20
            memory[8] = 24;
            memory[12] = 100;
            let foo: i32 = read(0, 8, 1, 4);

            let hello: ImmutableString = "Hello ";
            let i: i32 = 1000;
            for c in hello {
                memory[i] = c;
                i += 1;
            }
            // put exlamation mark at memory[500]
            memory[500] = '!';

            // store hello to iovectors
            memory[0] = 1000;
            memory[4] = i;
            // memory[8] and memory[20] already have the read vector
            // add vector for the exlamation mark
            memory[16] = 500;
            memory[20] = 1;

            // `let: foo`` is small hack, if a function returns a value it needs to be somehow consumed
            let foo: i32 = write(
                1, // stdout
                0, // io vectors start
                3, // number of io vectors
                50, // where to write the result
            );
        }
    };

    println!("{module}");
}
