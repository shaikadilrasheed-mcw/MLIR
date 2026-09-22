include "mlir/Pass/PassBase.td"

def CustomToArithPass : Pass<"custom-to-arith", "::mlir::ModuleOp"> {
    let summary = "Lower numeric dialect to arith dialect";
    let description = [{
        Lowers custom.add, custom.sub, custom.const, and custom.mul
        to their arith dialect equivalents.
    }];
}

def CustomToTosaPass : Pass<"custom-to-tosa", "::mlir::ModuleOp"> {
    let summary = "Lower numeric dialect to arith dialect";
    let description = [{
        Lowers custom.add, custom.sub to their tosa dialect equivalents.
    }];
}

def CustomToLoopsPass : Pass<"custom-to-loops", "::mlir::ModuleOp"> {
    let summary = "Lower custom.cumsum to scf loops + arith";
}
