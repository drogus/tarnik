#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod tests {
    use crate::test_helpers::TestRunner;
    use tarnik::wasm;
    use tarnik_ast::WatModule;

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
