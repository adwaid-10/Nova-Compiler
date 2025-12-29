// LLVM includes
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  // Manual argument parsing to avoid duplicate LLVM option registration
  std::string InputFilename = "-";
  std::string OutputFilename = "";
  bool Help = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      Help = true;
    } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
      OutputFilename = argv[++i];
    } else if (arg[0] != '-') {
      InputFilename = arg;
    }
  }

  if (Help) {
    llvm::errs() << "Usage: nova-translate <input.mlir> [-o <output.o>]\n";
    return 0;
  }

  // Register only the minimal required dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::LLVM::LLVMDialect, mlir::BuiltinDialect>();
  mlir::MLIRContext context(registry);

  // Load the MLIR module
  std::string errorMessage;
  auto file = openInputFile(InputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening input file: " << errorMessage << "\n";
    return 1;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<Operation *> module = parseSourceFile(*sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error parsing MLIR file.\n";
    return 1;
  }

  // Translate to LLVM IR
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR to LLVM IR.\n";
    return 1;
  }

  // Initialize LLVM targets
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  llvmModule->setTargetTriple(targetTriple);

  std::string error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple.getTriple(), error);
  if (!target) {
    llvm::errs() << "Error looking up target: " << error << "\n";
    return 1;
  }

  llvm::TargetOptions opt;
  auto RM = std::optional<llvm::Reloc::Model>();
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, "generic", "", opt, RM));
  
  if (!targetMachine) {
    llvm::errs() << "Error creating target machine.\n";
    return 1;
  }
  
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  // Prepare output stream
  std::error_code EC;
  llvm::raw_fd_ostream out(OutputFilename.empty() ? "a.o" : OutputFilename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Could not open output file: " << EC.message() << "\n";
    return 1;
  }

  // Emit object file
  llvm::legacy::PassManager pm;
  if (targetMachine->addPassesToEmitFile(pm, out, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "TargetMachine can't emit an object file.\n";
    return 1;
  }

  pm.run(*llvmModule);

  return 0;
}
