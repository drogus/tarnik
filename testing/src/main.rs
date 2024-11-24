use wazap_generator::wasm;

fn main() {
    let module = wasm! {
         type String = [mut i8];

        fn run() {
            let str: String = "foo";
            str[2] = 'i';
            if str[0] == 'f' {

            }
        }
    };

    println!("{module}");
}
