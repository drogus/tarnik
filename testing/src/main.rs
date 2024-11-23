use wazap_generator::wasm;

fn main() {
    let module = wasm! {
        type I32Array = [mut i32];

        fn sum(arr: I32Array) -> i32 {
            let result: i32 = 0;
            for element in arr {
                result += element;
            }

            return result;
        }

        fn run() {
            let arr: I32Array = [1, 2, 3];
            let s: i32 = sum(arr);
        }
    };

    println!("{module}");
}
