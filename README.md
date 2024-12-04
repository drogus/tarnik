## Tarnik

Tarnik provides a Rust macro that generates WASM GC code

### Why?

When you compile low level languages like C or Rust to WASM, you
get a WASM binary using only WebAssembly core instructions. I haven't
found a good way to generate a higher level WebAssembly code that uses
WASM GC features like reference types.

I have started working on it to help with [jawsm](https://github.com/drogus/jawsm)
development.

### What's up with the name?

I wanted to call the project Rasp, but it was already taken on crates.io,
so I used tye Polish translation, which surprisingly isn't a non-pronounceable word.
Yes, I am lazy.

### How do I use it?

After adding the crate to a Rust project you can do the following:

```rust

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
```

If you run such a program and save the output to a file:

```
cargo run > hello.wat
```

And compile:

```
wasm-tools parse hello.wat -o hello.wasm
```

You can run it with a runtime that supports WASM GC and WASIp1:

```
wasmedge run --enable-gc print.wasm
Hello world!
```

As this is an early version, a lot of expressions don't work correclty. I'll be
adding more stuff in the near future. The code used in the macro is "almost Rust",
ie. most of the syntax is taken from Rust, but it differs in some places where it made
sense, like defining an array with mutable fields. For a few more examples you can
take a look at [tests](testing/src/lib.rs).

If you want to speed up the development, please consider [sponsoring my work](https://github.com/sponsors/drogus)

### Roadmap

I don't have a long term plan for Tarnik, but I would really like to start using
it for jawsm, and thus I plan to add the following features soon-ish:

1. Support for defining and accessing memory. For example `@memory[0]` would read
   a value from memory named `$memory`. `@memory[0] = 1` would save the value into
   memory. I would probably also add some range support
2. Implement more binary and unary operators. For example equality only works for numeric
   types at the moment and a lot of operators don't work at all
3. Invest some time in error messages - some error mesasges will not show where
   the error is
4. Implement some built-in functions for manipulating data like copying an array range
   into memory or copying a memory range into an array. Also copying between arrays
   and maybe other operators/functions like that so increasing arrays size is easier
5. Implement support for try/catch and throw

### License

The project is licensed under Apache 2.0 license.
