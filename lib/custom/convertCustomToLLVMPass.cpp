#include "custom/customDialect.h"
#include "custom/customOps.h"
#include "custom/customPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <memory>

// Pull in the generated base class for CustomToLLVM
#define GEN_PASS_DEF_CUSTOMTOLLVM
#include "custom/customPasses.h.inc"

namespace {
struct ConvertAddCMul : public mlir::OpConversionPattern<custom::AddCMul> {
  using mlir::OpConversionPattern<custom::AddCMul>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(custom::AddCMul op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value a = adaptor.getA();
    mlir::Value b = adaptor.getB();
    mlir::Value c = adaptor.getC();

    // ---- SCALAR case ----
    if (!mlir::isa<mlir::RankedTensorType>(a.getType())) {
      mlir::Value prod = rewriter.create<mlir::arith::MulIOp>(loc, b, c);
      mlir::Value out  = rewriter.create<mlir::arith::AddIOp>(loc, a, prod);
      rewriter.replaceOp(op, out);
      return mlir::success();
    }

    // ---- TENSOR case ----
      mlir::Type type = a.getType();
      auto shiftTy = mlir::RankedTensorType::get({1},rewriter.getI8Type());
      mlir::Value shift = rewriter.create<mlir::tosa::ConstOp>(loc,shiftTy,mlir::DenseElementsAttr::get(shiftTy,(int8_t)0));
      mlir::Value prod = rewriter.create<mlir::tosa::MulOp>(loc, type, b, c, shift);
      mlir::Value out  = rewriter.create<mlir::tosa::AddOp>(loc, type, a, prod);
      rewriter.replaceOp(op, out);
      return mlir::success();

  }
};
} // namespace

namespace custom {

class CustomToLLVM : public impl::CustomToLLVMBase<CustomToLLVM> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::tosa::TosaDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                           mlir::scf::SCFDialect, mlir::tensor::TensorDialect,mlir::tosa::TosaDialect>();
    target.addIllegalDialect<custom::CustomDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAddCMul>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createCustomToLLVM() {
  return std::make_unique<CustomToLLVM>();
}
} // namespace custom