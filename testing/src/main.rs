fn main() {
    let module: tarnik_ast::WatModule = tarnik::wasm! {
        #[export("memory")]
        memory!("memory", 1);

        #[import("wasi_snapshot_preview1", "fd_write")]
        fn write(a1: i32, a2: i32, a3: i32, a4: i32) -> i32;

        type ImmutableString = [i8];

        #[export("_start")]
        fn run() {
            let str: ImmutableString = "Hello world!";
            let i: i32 = 100;
            for c in str {
                memory[i] = c;
                i += 1;
            }
            // store io vectors
            memory[0] = 100;
            memory[4] = i;

            // `let: foo`` is small hack, if a function returns a value it needs to be somehow consumed
            let foo: i32 = write(
                1, // stdout
                0, // io vectors start
                1, // number of io vectors
                50, // where to write the result
            );
        }
    };

    println!("{module}");
}
