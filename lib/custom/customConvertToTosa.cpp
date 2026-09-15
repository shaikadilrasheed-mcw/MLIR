#include "custom/customDialect.h"
#include "custom/customOps.h"
#include "custom/customPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

// Pull in the generated base class for CustomToTosaPass
#define GEN_PASS_DEF_CUSTOMTOTOSAPASS
#include "custom/customPasses.h.inc"

namespace {
struct CustomToTosaAddOp : public mlir::OpConversionPattern<custom::addOp> {
  using mlir::OpConversionPattern<custom::addOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::addOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op, op.getResult().getType(),
                                                   lhs, rhs);
    return mlir::success();
  }
};

struct CustomToTosaSubOp : public mlir::OpConversionPattern<custom::subOp> {
  using mlir::OpConversionPattern<custom::subOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::subOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<mlir::tosa::SubOp>(op, op.getResult().getType(),
                                                   lhs, rhs);
    return mlir::success();
  }
};
} // namespace

namespace custom {
class CustomToTosaPass : public impl::CustomToTosaPassBase<CustomToTosaPass> {
public:
  // Tell MLIR which dialects this pass introduces
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::tosa::TosaDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    // After conversion: Tosa ops are LEGAL
    target.addLegalDialect<mlir::tosa::TosaDialect>();

    // After conversion: custom ops are ILLEGAL
    target.addIllegalDialect<custom::CustomDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    // custom.add → tosa.add
    patterns.add<CustomToTosaAddOp>(&getContext());
    // custom.sub → tosa.sub
    patterns.add<CustomToTosaSubOp>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
std::unique_ptr<mlir::Pass> createCustomToTosaPass() {
  return std::make_unique<CustomToTosaPass>();
}
} // namespace custom