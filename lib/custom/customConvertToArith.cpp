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

// Pull in the generated base class for CustomToArithPass
#define GEN_PASS_DEF_CUSTOMTOARITHPASS
#include "custom/customPasses.h.inc"

namespace {
struct ConvertAddOp : public mlir::OpConversionPattern<custom::addOp> {
  using mlir::OpConversionPattern<custom::addOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::addOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, lhs, rhs);
    return mlir::success();
  }
};
struct ConvertSubOp : public mlir::OpConversionPattern<custom::subOp> {
  using mlir::OpConversionPattern<custom::subOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::subOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, lhs, rhs);
    return mlir::success();
  }
};
struct ConvertMulOp : public mlir::OpConversionPattern<custom::mulOp> {
  using mlir::OpConversionPattern<custom::mulOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::mulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, lhs, rhs);
    return mlir::success();
  }
};
struct ConvertConstantOp : public mlir::OpConversionPattern<custom::constant> {
  using mlir::OpConversionPattern<custom::constant>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::constant op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.getValueAttr());
    return mlir::success();
  }
};
} // namespace

namespace custom {

class CustomToArithPass
    : public impl::CustomToArithPassBase<CustomToArithPass> {
public:
  // Tell MLIR which dialects this pass introduces
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::tosa::TosaDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    // After conversion: arith, func ops are LEGAL
    target
        .addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect>();

    // After conversion: custom ops are ILLEGAL
    target.addIllegalDialect<custom::CustomDialect>();

    mlir::RewritePatternSet patterns(&getContext());

    // Register patterns using the template:
    // custom.add → arith.addi
    patterns.add<ConvertAddOp>(&getContext());
    // custom.sub → arith.subi
    patterns.add<ConvertSubOp>(&getContext());
    // custom.mul → arith.muli
    patterns.add<ConvertMulOp>(&getContext());
    // custom.constant → arith.constant
    patterns.add<ConvertConstantOp>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};
std::unique_ptr<mlir::Pass> createCustomToArithPass() {
  return std::make_unique<CustomToArithPass>();
}
} // namespace custom