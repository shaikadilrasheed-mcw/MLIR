#include "custom-c/Passes.h"
#include "custom/customPasses.h"

#include "mlir/Conversion/Passes.h"
// ... other MLIR pass headers ...

void customRegisterAllPasses(void) {
  mlir::registerConvertMathToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  // ... etc.
  custom::registerCustomPasses();
}