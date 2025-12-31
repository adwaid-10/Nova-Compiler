//===- CudaKernels.cpp - GPU Kernel Wrappers ------------------------------===//
//
// Nova Compiler Runtime
//
//===----------------------------------------------------------------------===//

#include "Runtime/Kernels/KernelRegistration.h"
#include "Runtime/Core/AsyncValue.h"
#include "Runtime/Core/HostContext.h"

namespace nova {
namespace runtime {

void RegisterCudaKernels(KernelRegistry& registry) {
  // TODO: Implement CUDA wrappers similar to HostKernels
  // registry.RegisterKernel("nova.add", Device::kGPU, CudaAddWrapper);
}

void RegisterAllKernels(KernelRegistry& registry) {
  RegisterHostKernels(registry);
  RegisterCudaKernels(registry);
}

} // namespace runtime
} // namespace nova
