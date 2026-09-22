#include "custom/customDialect.h"
#include "custom/customOps.h"
#include "custom/customPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h" 
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

// Pull in the generated base class for CustomToArithPass
#define GEN_PASS_DEF_CUSTOMTOLOOPSPASS
#include "custom/customPasses.h.inc"

struct ConvertCumsum : public mlir::OpConversionPattern<custom::cumsumOp> {
  using mlir::OpConversionPattern<custom::cumsumOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(custom::cumsumOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value tensor = adaptor.getTensor();          // the input tensor
    mlir::Type elemTy  = op.getResult().getType();     // i64

    // --- constants ---
    mlir::Value c0   = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1   = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, elemTy, 0);

    // --- N = size of the tensor's dim 0 ---
    mlir::Value n = rewriter.create<mlir::tensor::DimOp>(loc, tensor, c0);

    // --- scf.for with an accumulator (iter_arg) ---
    auto loop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, n, c1, mlir::ValueRange{zero},
        [&](mlir::OpBuilder &b, mlir::Location l,
            mlir::Value iv, mlir::ValueRange args) {
          mlir::Value acc = args[0];                                  // running sum
          mlir::Value v   = b.create<mlir::tensor::ExtractOp>(
                                l, tensor, mlir::ValueRange{iv});      // tensor[i]
          mlir::Value sum = b.create<mlir::arith::AddIOp>(l, acc, v); // acc + v
          b.create<mlir::scf::YieldOp>(l, sum);                       // pass to next iter
        });

    // --- the loop's result replaces the custom.cumsum ---
    rewriter.replaceOp(op, loop.getResult(0));
    return mlir::success();
  }
};



namespace custom {

class CustomToLoopsPass
    : public impl::CustomToLoopsPassBase<CustomToLoopsPass> {
public:
  // Tell MLIR which dialects this pass introduces
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    // After conversion: arith, func ops are LEGAL
    target
        .addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,mlir::scf::SCFDialect,mlir::tensor::TensorDialect>();

    // After conversion: custom ops are ILLEGAL
    target.addIllegalDialect<custom::CustomDialect>();

    mlir::RewritePatternSet patterns(&getContext());

    // Register patterns using the template:
    // custom.constant → scf.for
    patterns.add<ConvertCumsum>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
std::unique_ptr<mlir::Pass> createCustomToLoopsPass() {
  return std::make_unique<CustomToLoopsPass>();
}
} // namespace custom