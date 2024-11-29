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
let module = tarnik::wasm! {
  type String = [mut i8];

  fn run() {
    let foo: String = "Hello world";
    foo[1] = 'a';

    let sum: i32 = 0;
    for byte in foo {
      sum += byte;
    }
  }
};

println!("{module}");
```

In order to make the module a bit more useful, you may have to add some stuff like memory or imports.
I plan to add APIs for that in the macro itself, but at the moment the easiest way to do that is to
modify the module struct, like so:

```rust
let mut module: WatModule = wasm! {
    fn run() {
        let x: i32 = 0;
    }
};

module.add_memory("$memory", 1);
module.add_export("memory", "memory", "$memory");
module.add_export("_start", "func", "$run");
module.add_import(
    "console",
    "log",
    WasmType::func(vec![WasmType::I32, WasmType::I32], None),
);

println!("{module}");
```

This will result in the following WAT program:

```
(module
  (import "console" "log" (func (param i32) (param i32)))

  (memory $memory 1)

  (elem declare func $run)
  (func $run
    (local $x i32)
    (i32.const 0)
    (local.set $x)
  )

  (export "memory" (memory $memory))
  (export "_start" (func $run))
)
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
