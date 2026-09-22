#ifndef CUS_C_DIALECTS_H
#define CUS_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#ifdef __cplusplus
extern "C" { 
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Custom, custom);

MLIR_CAPI_EXPORTED void customRegisterAllExtensions(MlirDialectRegistry registry);
#ifdef __cplusplus
}
#endif

#endif // CUS_C_DIALECTS_H