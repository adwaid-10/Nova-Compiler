//===- HostKernels.cpp - CPU Kernel Wrappers ------------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Kernels/KernelRegistration.h"
#include "Runtime/Core/AsyncValue.h"
#include "Runtime/Core/HostContext.h"
#include <iostream>

// --- TensorLib Integration ---
// Since TensorLib might be linked externally, we assume headers are available.
// If not available during this build step, we'll mock the operations for the scaffold.
#if __has_include("core/Tensor.h") || defined(HAS_TENSORLIB)
#include "core/Tensor.h"
#include "ops/TensorOps.h"
using Tensor = OwnTensor::Tensor;
#else
// Mock Tensor for standalone compilation if headers missing
namespace OwnTensor {
  struct Tensor { 
    // minimal mock
    static Tensor zeros(std::vector<long>, bool) { return {}; }
  }; 
}
using Tensor = OwnTensor::Tensor;
#endif

namespace nova {
namespace runtime {

// --- Helper: AsyncValue Unwrap/Wrap ---

// Example Wrapper for "nova.add"
// Inputs: [AsyncValue<Tensor>, AsyncValue<Tensor>]
// Output: [AsyncValue<Tensor>]
AsyncValue* AddWrapper(const std::vector<AsyncValue*>& args, HostContext* host) {
  // 1. Unwrap Inputs (Blocking for simplicity in this wrapper, 
  //    but practically they are already "Available" effectively if dispatched)
  //    In a pure async world, we might register a callback. 
  //    But ExecutionEngine usually ensures inputs are ready before dispatching this closure.
  
  // Cast raw void* back to Tensor* (Assuming we stored Tensor* in the AsyncValue)
  // Phase 2 ResolveArg stored `void*` -> we cast to Tensor*
  
  if (args.size() != 2) return host->MakeErrorAsyncValue("Add requires 2 arguments");
  
  // NOTE: This assumes the AsyncValues contain `Tensor*` or `Tensor`.
  // Our ExecutionEngine stores `void*`. 
  // Let's assume the void* points to a Tensor object managed elsewhere or passed in.
  
  auto* lhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[0]);
  auto* rhs_av = dynamic_cast<ConcreteAsyncValue<void*>*>(args[1]);
  
  if (!lhs_av || !rhs_av) return host->MakeErrorAsyncValue("Invalid arguments");
  
  Tensor* lhs = static_cast<Tensor*>(lhs_av->get());
  Tensor* rhs = static_cast<Tensor*>(rhs_av->get());
  
  // 2. Call Native Kernel
  // Using OwnTensor operator overload or function
  // Tensor result = *lhs + *rhs; 
  // For safety without full headers, let's just print or mock if headers missing
  
  // Silence unused variable warnings for scaffold
  (void)lhs; 
  (void)rhs;

#if defined(HAS_TENSORLIB)
  // Actual call
   Tensor* result_tensor = new Tensor(*lhs + *rhs);
#else
  // Scaffold call
  // std::cout << "Executing Host Add (Mock)\n";
  Tensor* result_tensor = new Tensor(); 
#endif

  // 3. Wrap Result
  return host->MakeAvailableAsyncValue<void*>(result_tensor);
}

// --- Registration ---

void RegisterHostKernels(KernelRegistry& registry) {
  registry.RegisterKernel("nova.add", Device::CPU, AddWrapper);
  // Add other kernels...
}

} // namespace runtime
} // namespace nova
