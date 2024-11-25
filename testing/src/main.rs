use wazap_generator::wasm;

fn main() {
    let module = wasm! {
         type String = [mut i8];

        fn run() {
            let str: String;
            str = "foo";
            str[2] = 'i';
            assert(1, "This should not throw");
            assert(0, "This should throw");
        }
    };

    println!("{module}");
}
