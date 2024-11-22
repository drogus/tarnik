use wazap_generator::wasm;

fn main() {
    let foo = wasm! {
        fn foo(a: i32, b: i32) -> i32 {
            return a + b;
        }
    };

    println!("{foo}");
}
