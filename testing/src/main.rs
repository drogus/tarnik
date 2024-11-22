use wazap_generator::wasm;

fn main() {
    let foo = wasm! {
        type I32Array = [mut i32];

        fn foo(a: i32, b: i32) -> i32 {
            let foo: Nullable<I32Array> = [0; 3];
            return a + b;
        }
    };

    println!("{foo}");
}
