#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod tests {
    use crate::test_helpers::TestRunner;
    use tarnik::wasm;
    use tarnik_ast::WatModule;

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
