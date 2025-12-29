# Deep Dive: Generating Binary Code from MLIR/LLVM IR

This document explains the technical implementation and theory of how the **Nova Compiler** transforms high-level MLIR IR into hardware-executable binary code for both CPUs and NVIDIA GPUs.

---

## 1. The Theory: From Dialects to Data

The transformation follows a strict hierarchy of lowering:
1.  **MLIR Dialects** (`nova`, `linalg`, `affine`...) $\rightarrow$ **LLVM Dialect** (within MLIR).
2.  **LLVM Dialect** $\rightarrow$ **LLVM IR** (Bitcode/Text).
3.  **LLVM IR** $\rightarrow$ **Target Assembly/PTX** (Using Target Machines).
4.  **Assembly** $\rightarrow$ **Machine Code** (Binary Object/CUBIN).

---

## 2. The CPU Path: MLIR to Object File

For the CPU, we treat the code as if we were a standard compiler (like Clang).

### Theory
We use the **LLVM TargetMachine** to convert IR into machine-specific instructions. The `TargetMachine` understands the CPU's Instruction Set Architecture (ISA), like x86-64 or ARM.

### Code Implementation (C++)

```cpp
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

void generateCPUObject(llvm::Module &module, std::string filename) {
    // 1. Initialize LLVM Targets
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    // 2. Define the Target (e.g., "x86_64-pc-linux-gnu")
    auto TargetTriple = llvm::sys::getDefaultTargetTriple();
    std::string Error;
    auto Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);

    // 3. Create the Target Machine
    auto CPU = "generic";
    auto Features = "";
    llvm::TargetOptions opt;
    auto RM = std::optional<llvm::Reloc::Model>();
    auto TargetMachine = Target->createTargetMachine(TargetTriple, CPU, Features, opt, RM);

    // 4. Output the Object File
    std::error_code EC;
    llvm::raw_fd_ostream dest(filename, EC, llvm::sys::fs::OF_None);
    
    llvm::legacy::PassManager pass;
    auto FileType = llvm::CodeGenFileType::CGFT_ObjectFile;
    TargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType);
    pass.run(module);
    dest.flush();
}
```

---

## 3. The GPU Path: NVVM to PTX/CUBIN

GPU compilation is "Special" because NVIDIA GPUs don't consume standard ELF/EXE files; they consume **PTX** (virtual assembly) or **CUBIN** (binary).

### Theory
Instead of the standard x86 target, we use the **NVPTX** (NVIDIA PTX) backend. This converts NVVM IR (which contains GPU-specific intrinsics like `llvm.nvvm.read.ptx.sreg.tid.x`) into textual PTX code.

### Code Implementation (C++)

```cpp
#include "llvm/Target/TargetMachine.h"

std::string generatePTX(llvm::Module &module) {
    // 1. Initialize NVPTX Target
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    // 2. Setup Target Architecture (e.g., nvptx64-nvidia-cuda)
    auto Triple = "nvptx64-nvidia-cuda";
    std::string Error;
    auto Target = llvm::TargetRegistry::lookupTarget(Triple, Error);

    // 3. Setup GPU Features (sm_86 for RTX 30-series, etc.)
    auto CPU = "sm_86"; 
    auto TargetMachine = Target->createTargetMachine(Triple, CPU, "", {}, {});

    // 4. Emit PTX as a String
    std::string ptx;
    llvm::raw_string_ostream stream(ptx);
    llvm::legacy::PassManager pm;
    TargetMachine->addPassesToEmitFile(pm, stream, nullptr, llvm::CodeGenFileType::CGFT_AssemblyFile);
    pm.run(module);
    
    return ptx;
}
```

### The Final Step: NVRTC (JIT)
The generated **PTX string** is then handed to the NVIDIA Runtime Compiler (`nvrtc`) at execution time to produce the final binary code (**CUBIN**) for the user's specific GPU hardware.

---

## 4. Summary of the Bridge

| Component | CPU Strategy | GPU Strategy |
| :--- | :--- | :--- |
| **LLVM Backend** | `X86_64` / `AArch64` | `NVPTX` |
| **Output Format** | `.obj` / `.o` (Binary) | `PTX` (Textual IR) |
| **Linker** | `lld` (Static Linker) | `nvrtc` (JIT Runtime Compiler) |
| **Execution** | `dlopen` + Function Pointer | `cuModuleLoad` + `cuLaunchKernel` |

---

## 5. Why this matters for Nova
By using the LLVM APIs directly:
1.  **Optimization**: LLVM performs machine-specific optimizations (AVX-512, TensorCore utilization) that are impossible at the MLIR level.
2.  **Portability**: We can target any CPU or GPU without rewriting the compiler—just by changing the `TargetTriple`.
3.  **AOT vs JIT**: We can compile models to disk (Ahead-Of-Time) for deployment or compile them in memory (Just-In-Time) for interactive notebooks.
