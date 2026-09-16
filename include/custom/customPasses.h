#ifndef CUSTOM_PASSES_H
#define CUSTOM_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace custom {

#define GEN_PASS_DECL
#include "custom/customPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "custom/customPasses.h.inc"

} // namespace custom

#endif
