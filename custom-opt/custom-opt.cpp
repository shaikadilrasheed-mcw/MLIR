#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"
#include "custom/customDialect.h"
#include "custom/customOps.h"
#include "mlir/Pass/PassRegistry.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<custom::CustomDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::tosa::TosaDialect,mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::affine::AffineDialect
                    >();
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "custom optimizer\n", registry));
}
