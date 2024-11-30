const fs = require("fs");

async function runWasm(wasmPath) {
  const wasmBuffer = fs.readFileSync(wasmPath);

  let memory;
  const assertExceptionTag = new WebAssembly.Tag({
    parameters: ["i32", "i32"],
  });

  const importObject = {
    env: {
      assert_exception_tag: assertExceptionTag,
    },
    console: {
      log: (offset, length) => {
        const bytes = new Uint8Array(memory.buffer, offset, length);
        const text = new TextDecoder().decode(bytes);
        process.stdout.write(text);
      },
    },
  };

  try {
    const wasmModule = await WebAssembly.instantiate(wasmBuffer, importObject);
    const instance = wasmModule.instance;

    memory = instance.exports.memory;

    // Call the _start function if it exists
    if (instance.exports._start) {
      try {
        instance.exports._start();
      } catch (e) {
        if (e.is(assertExceptionTag)) {
          let offset = e.getArg(assertExceptionTag, 0);
          let length = e.getArg(assertExceptionTag, 1);

          const bytes = new Uint8Array(memory.buffer, offset, length);
          const text = new TextDecoder().decode(bytes);
          console.error("Assertion failed:", text);
        } else {
          console.log("Unknown error thrown", e);
        }
      }
    } else {
      console.error("Could not find _start function");
    }

    return { success: true };
  } catch (error) {
    if (error instanceof WebAssembly.RuntimeError) {
      return { success: false, error: error.message };
    }
    throw error;
  }
}

// If run directly from command line
if (require.main === module) {
  const wasmPath = process.argv[2];
  if (!wasmPath) {
    console.error("Please provide a path to a WASM file");
    process.exit(1);
  }

  runWasm(wasmPath)
    .then((result) => {
      if (!result.success) {
        process.exit(1);
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      process.exit(1);
    });
}
