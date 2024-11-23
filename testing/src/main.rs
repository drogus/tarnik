use wazap_generator::wasm;

fn main() {
    let module = wasm! {
         struct Foo {
            a: mut i32,
            b: i32
        }

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
            let b: i32 = arr[1];
            arr[2] = 10;
            let s: i32 = sum(arr);
        }
    };

    println!("{module}");
}
