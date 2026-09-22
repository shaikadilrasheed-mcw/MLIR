#include "custom-c/Passes.h"
#include "custom/customPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"   // for registerExpandStridedMetadataPass
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/InitAllExtensions.h"
// ... other MLIR pass headers ...

void customRegisterAllPasses(void) {
  mlir::registerConvertMathToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  // ... etc.
  mlir::registerSCFToControlFlowPass();              // -> "convert-scf-to-cf"
  mlir::registerConvertControlFlowToLLVMPass();      // -> "convert-cf-to-llvm"
  mlir::registerFinalizeMemRefToLLVMConversionPass();// -> "finalize-memref-to-llvm"
  mlir::memref::registerExpandStridedMetadataPass(); // -> "expand-strided-metadata"
  mlir::bufferization::registerBufferizationPasses();
  custom::registerCustomPasses();
}