#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "custom/customDialect.h"
#include "custom/customOps.h"

using namespace mlir;
using namespace custom;

//===----------------------------------------------------------------------===//
// custom dialect.
//===----------------------------------------------------------------------===//

#include "custom/customOpsDialect.cpp.inc"

void CustomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "custom/customOps.cpp.inc"
      >();
}
