include "customDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

def custom_addOp : Custom_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "Adds two integer,tensor values";
    let arguments = (ins AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $lhs, AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $rhs );
    let results   = (outs AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def custom_subOp : Custom_Op<"sub", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "This Op Performs subtraction";
    let arguments = (ins AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $lhs, AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $rhs );
    let results   = (outs AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def custom_mulOp : Custom_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
    let summary = "This Op Performs multiplication";
    let arguments = (ins AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $lhs, AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $rhs );
    let results   = (outs AnyTypeOf<[AnyInteger,TensorOf<[AnyInteger]>]> : $result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def custom_constant : Custom_Op<"constant", [Pure]> {
    let summary = "constant value";
    let arguments = (ins I64Attr:$value);
    let results   = (outs I64:$result);
    let assemblyFormat = "$value attr-dict `:` type($result)";
}
