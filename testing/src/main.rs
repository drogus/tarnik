use tarnik_testing::test_helpers;

#[cfg(test)]
mod tests {
    use super::test_helpers::TestRunner;
    use tarnik::wasm;
    use tarnik_ast::WatModule;

    #[test]
    fn test_ref_equality() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            memory!("memory", 1);

            struct Foo {}

            fn run() {
                let foo_1: Foo = Foo {};
                let foo_2: Foo = foo_1;
                let foo_3: Foo = Foo {};
                let foo_4: Foo = foo_3;

                assert(foo_1 == foo_2, "the references foo_1 and foo_2 should match");
                assert(foo_3 == foo_4, "the references foo_3 and foo_4 should match");
                assert(foo_1 != foo_3, "the references foo_1 and foo_3 should not match");
                assert(foo_1 != foo_4, "the references foo_1 and foo_4 should not match");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_data() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            memory!("memory", 1);

            fn run() {
                let offset1: i32 = data!("foo");
                let offset2: i32 = data!("bar");
                let offset3: i32 = data!("foo");

                assert(offset1 == offset3, "data entries should not add a new entry if one exists");
                assert(memory[offset2] == 'b', "offset should point at the data");
                assert(memory[offset2 + 1] == 'a', "offset should point at the data");
                assert(memory[offset2 + 2] == 'r', "offset should point at the data");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_ref_func() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type FooFunc = fn(i32) -> i32;

            fn run() {
                let foo_ref: FooFunc = foo;

                let y: i32 = foo_ref(33);

                assert(y == 44, "it should be possible to call a func ref");
            }

            fn foo(x: i32) -> i32 {
                return x + 11;
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_global_array() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type Arr = [Nullable<anyref>];
            static foo: Arr = [null; 10];

            fn run() {
                let first_element: anyref = foo[0];
                let is_null: i32 = ref_test!(first_element, null);
                assert(is_null == 1, "array should be initialized and values should be null");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_global_null() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            struct Foo {}
            static foo: Nullable<Foo> = null;

            fn run() {
                let is_null: i32 = ref_test!(foo, null);
                assert(is_null == 1, "global should be set to null");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_global() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            static mut foo: i64 = 10;

            fn run() {
                foo += 34;
                assert(foo == 44, "global foo should");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_try_catch() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type ExceptionType = fn(i32, i64);
            tag!(Exception, ExceptionType);

            fn run() {
                let a: i32 = 0;
                let outer_x: i32;
                let outer_y: i64;

                try {
                    a += 10;
                    throw!(Exception, 1, 2 as i64);
                }
                catch(Exception, x: i32, y: i64) {
                    a += 100;
                    outer_x = x;
                    outer_y = y;
                }
                catch_all {
                    a += 1000;
                }
                assert(outer_x == 1, "first argument to the exception should be equal to 1");
                assert(outer_y == 2, "second argument to the exception should be equal to 2");
                assert(a == 1110, "a should equal to 1110");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_nullable() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type I64Array = [i64];
            fn run() {
                let x: Nullable<I64Array> = null;
                // TODO: refactor when negate is implemented
                if ref_test!(x, null) {
                } else {
                    assert(0, "Couldn't successfully test for null");
                }
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_len() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type I64Array = [i64];
            fn run() {
                let x: I64Array = [44, 10, 11, 12];
                assert(len!(x) == 4, "The $x array should have 4 elements");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_casting() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type I64Array = [i64];
            fn run() {
                let x: I64Array = [44];
                test_any(x);
            }

            fn test_any(numbers: anyref) {
                if ref_test!(numbers, I64Array) {
                    let numbers_i64: I64Array = numbers as I64Array;
                    assert(numbers_i64[0] == 44, "The first element of the numbers array should be 44");
                } else {
                    assert(0, "$numbers should be an I64Array");
                }
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_arithmetic_operations() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            fn run() {
                let x: i32 = 0;
                assert(x + 1 == 1, "0 + 1 should equal 1");
                assert(x - 1 == -1, "0 - 1 should equal -1");
                x = 4;
                assert(x * 4 == 16, "4 * 4 should equal 16");
                assert(x / 2 == 2, "4 / 1 should equal 2");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_array_operations() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type Array = [mut i32];

            fn run() {
                let arr: Array;
                arr = [1, 2, 3];
                arr[1] = 42;
                assert(arr[1] == 42, "Array modification failed");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_string_operations() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type String = [mut i8];

            fn run() {
                let str: String;
                str = "hello";
                str[1] = 'a';
                assert(str[1] == 'a', "String modification failed");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_for_loop() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            type Array = [mut i32];

            fn run() {
                let arr: Array = [1, 2, 3];
                let sum: i32 = 0;
                for x in arr {
                    sum = sum + x;
                }
                assert(sum == 6, "For loop sum incorrect");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_while_loop() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            fn run() {
                let i: i32 = 0;
                let j: i32 = 0;

                while i < 10 {
                    if j < 3 {
                        j += 1;
                        continue;
                    }

                    if i == 5 {
                        break;
                    }

                    i += 1;
                    j += 1;
                }
                assert(i == 5, "break works properly");
                assert(j == 8, "continue works properly");
            }
        };

        // println!("module: {module:?}");
        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_if_statement() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            fn run() {
                let x: i32 = 42;
                if x == 42 {
                    assert(1, "If statement worked");
                } else {
                    x = 10;
                }

                let y: bool = x == 10;
                assert(y, "Else branch should be executed");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }

    #[test]
    fn test_number_addition() -> anyhow::Result<()> {
        let module: WatModule = wasm! {
            fn run() {
                let x: i32 = 40;
                let y: i32 = 2;
                let z: i32 = x + y;
                assert(z == 42, "Addition failed");
            }
        };

        let runner = TestRunner::new()?;
        let (code, stdout, stderr) = runner.run_wasm_test(module)?;
        assert_eq!(code, 0, "stdout: {}\nstderr: {}", stdout, stderr);
        Ok(())
    }
}
