include "customDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def custom_addOp : Custom_Op<"add", [Pure, Commutative]> {
    let summary = "Adds two integer values";
    let description = [{ 
        This operation adds two integer values.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results   = (outs I64:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

