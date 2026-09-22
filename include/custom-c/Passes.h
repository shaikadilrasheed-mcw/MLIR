#ifndef CUS_C_PASSES_H
#define CUS_C_PASSES_H

#include "mlir-c/Support.h"
#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void customRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // CUS_C_PASSES_H