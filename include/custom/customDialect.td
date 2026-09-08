#ifndef CUSTOM_DIALECT
#define CUSTOM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// custom dialect definition.
//===----------------------------------------------------------------------===//

def Custom_Dialect : Dialect {
    let name = "custom";
    let summary = "A custom out-of-tree MLIR dialect.";
    let cppNamespace = "::custom";
}

//===----------------------------------------------------------------------===//
// Base custom operation definition.
//===----------------------------------------------------------------------===//

class Custom_Op<string mnemonic, list<Trait> traits = []> :
        Op<Custom_Dialect, mnemonic, traits>;

#endif // CUSTOM_DIALECT
