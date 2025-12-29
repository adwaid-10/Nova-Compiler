# Nova Compiler-to-Runtime Bridge (Simplified)

Your compiler stops at "Intermediate Code" (IR). To actually run it, we need two things:

## 1. The "Printer" (Binary Generation)
This takes the compiler's output and turns it into something the hardware can actually "read."

*   **For GPU**: It turns the IR into **PTX** (GPU instructions).
*   **For CPU**: It turns the IR into an **Object File** (a mini-program).

## 2. The "Driver" (Runtime Engine)
This is a C++ class in `cgadimpl` that acts as the "boss" to run the code.

1.  **Load**: It loads the instructions (PTX or shared library).
2.  **Plug in Data**: It takes the raw memory addresses from your `ag::Tensor` and "plugs" them into the instructions.
3.  **Go!**: It tells the hardware to start working (using CUDA for GPU or a direct call for CPU).

## Summary Table

| Hardware | What we generate | How we run it |
| :--- | :--- | :--- |
| **GPU** | PTX String | CUDA `cuLaunchKernel` |
| **CPU** | Shared Library | Direct function call |
