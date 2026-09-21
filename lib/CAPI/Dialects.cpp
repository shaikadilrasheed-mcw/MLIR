#include "custom-c/Dialects.h"
#include "custom/customDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"   
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
// ... other MLIR interface headers ...

void customRegisterAllExtensions(MlirDialectRegistry registry) {
  mlir::DialectRegistry *reg = unwrap(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(*reg);
  // ... etc.
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Custom, custom, custom::CustomDialect)