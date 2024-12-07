use anyhow::Result;
use std::fs;
use std::process::Command;
use tarnik_ast::{Signature, WasmType, WatModule};
use tempfile::TempDir;

pub struct TestRunner {
    temp_dir: TempDir,
}

impl TestRunner {
    pub fn new() -> Result<Self> {
        Ok(Self {
            temp_dir: TempDir::new()?,
        })
    }

    pub fn run_wasm_test(&self, module: WatModule) -> Result<(i32, String, String)> {
        // Add standard exports/imports
        let mut module = module;
        module.add_memory("$memory", 1, None);
        module.add_export("memory", "memory", "$memory");
        module.add_export("_start", "func", "$run");
        module.add_import(
            "env",
            "assert_exception_tag",
            WasmType::Tag {
                name: "$AssertException".to_string(),
                signature: Box::new(Signature {
                    params: vec![WasmType::I32, WasmType::I32],
                    result: None,
                }),
            },
        );

        // Write WAT file
        let wat_path = self.temp_dir.path().join("test.wat");
        fs::write(&wat_path, module.to_string())?;
        // println!("{module}");

        // Convert WAT to WASM
        let wasm_path = self.temp_dir.path().join("test.wasm");
        let status = Command::new("wasm-tools")
            .args([
                "parse",
                wat_path.to_str().unwrap(),
                "-o",
                wasm_path.to_str().unwrap(),
            ])
            .status()?;

        if !status.success() {
            anyhow::bail!("Failed to convert WAT to WASM");
        }

        // Run with Node.js
        let output = Command::new("node")
            .args(["run.js", wasm_path.to_str().unwrap()])
            .output()?;

        Ok((
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
        ))
    }
}
