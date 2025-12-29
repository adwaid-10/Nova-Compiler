# Internal Architecture: Nova-Translate Split & Merge

This document explains the technical "Magic" inside `nova-translate`—how it takes one MLIR file, splits it into two different hardware paths, and then merges them back into a single object file.

---

## 1. The Split: Kernel Outlining

The first step is a process called **Outlining**. 

### What happens?
In your MLIR code, you have a `gpu.launch` block. The compiler looks at this block and says: "This part doesn't belong on the CPU."
1.  It **extracts** the code inside the launch block.
2.  It creates a separate **GPU Module** (`gpu.module`) inside your module.
3.  It replaces the original block on the CPU side with a **Call** to a "Kernel Launcher."

### Visualization of the Split
```mlir
// BEFORE SPLIT
func.func @my_model(%arg0: tensor<f32>) {
  gpu.launch blocks(1,1,1) threads(32,1,1) {
    // Math logic for GPU
  }
}

// AFTER SPLIT (Outlined)
module {
  gpu.module @kernel_module {
    gpu.func @my_kernel(%arg0: memref<f32>) {
      // THE WORKER (GPU Code) 
    }
  }
  func.func @my_model(%arg0: memref<f32>) {
    gpu.launch_func @kernel_module::@my_kernel ...
    // THE BOSS (CPU Code)
  }
}
```

---

## 2. The Separate Compilation (The "Two Paths")

Now that the module is split, `nova-translate` runs **two separate compilation passes** at the same time. This is where the logic from `binary_code_generation.md` comes in.

### Path A: The Device (GPU)
*   It takes **only** the `gpu.module`.
*   It lowers it to **NVVM Dialect**.
*   It uses the `NVPTX` backend to generate **PTX String** (the code we discussed earlier).

### Path B: The Host (CPU)
*   It takes the **rest** of the module (the `func.func`).
*   It lowers it to **LLVM Dialect**.
*   It uses the `X86_64` backend to prepare **Object Code**.

---

## 3. The Merge: Binary Embedding

This is the clever part. We have a **PTX String** and **Object Code**. How do we merge them?

### The Method: Data Embedding
`nova-translate` creates a **Global Constant Variable** in the CPU's Object Code. It literally "copypastes" the entire PTX string into this variable.

```cpp
// SIMULATED OUTPUT CODE
// This lives inside your CPU Object file
const char* GPU_BINARY_DATA = "--- THE ENTIRE PTX CODE STRING GOES HERE ---";

void host_launcher() {
    // When this function runs, it knows to look at GPU_BINARY_DATA
    // and hand it to the CUDA driver.
    Runtime::launch(GPU_BINARY_DATA, ...);
}
```

---

## 4. Final Output: A Single `.nova` (or `.o`) File

When `nova-translate` finishes:
1.  **Logical Merge**: The GPU code is physically inside the CPU code as data.
2.  **One Binary**: You end up with **one single file** (usually an ELF/Object file).

### Why do we do it this way?
*   **Ease of Use**: You only have to manage one file for your model.
*   **Consistency**: The Host code and Device code are always in sync. You can't accidentally run Version 1 of the CPU code with Version 2 of the GPU code.

---

## 5. Connection to `binary_code_generation.md`

In my previous guide, I showed you `generateCPUObject()` and `generatePTX()`. 
*   In `nova-translate`, you call **both**. 
*   `generatePTX()` creates the "Letter."
*   `generateCPUObject()` creates the "Envelope" and places the "Letter" inside it.

This is the standard industry way to handle Heterogeneous computing (CUDA, OpenCL, SYCL).
