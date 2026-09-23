// File   : convertCustomToLoops.cpp
// Purpose: Lower custom dialect ops to arith/scf/tensor.
// Author : Adil
// Date   : 2026-09-24

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
#include <memory>

// Pull in the generated base class for CustomToLLVM
#define GEN_PASS_DEF_CUSTOMTOLOOPS
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

    // ---- SCALAR case: a + b*c ----
    if (!mlir::isa<mlir::RankedTensorType>(a.getType())) {
      mlir::Value prod = rewriter.create<mlir::arith::MulIOp>(loc, b, c);
      mlir::Value out  = rewriter.create<mlir::arith::AddIOp>(loc, a, prod);
      rewriter.replaceOp(op, out);
      return mlir::success();
    }

    // ---- TENSOR case: out[i] = a[i] + b[i]*c[i] ----
    auto tt = mlir::cast<mlir::RankedTensorType>(a.getType());
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value n  = rewriter.create<mlir::tensor::DimOp>(loc, a, c0);
    mlir::Value init = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tt.getShape(), tt.getElementType());

    auto loop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, n, c1, mlir::ValueRange{init},
        [&](mlir::OpBuilder &nb, mlir::Location nl, mlir::Value iv,
            mlir::ValueRange args) {
          mlir::Value acc = args[0];
          mlir::Value av = nb.create<mlir::tensor::ExtractOp>(nl, a, mlir::ValueRange{iv});
          mlir::Value bv = nb.create<mlir::tensor::ExtractOp>(nl, b, mlir::ValueRange{iv});
          mlir::Value cv = nb.create<mlir::tensor::ExtractOp>(nl, c, mlir::ValueRange{iv});
          mlir::Value prod = nb.create<mlir::arith::MulIOp>(nl, bv, cv);
          mlir::Value out  = nb.create<mlir::arith::AddIOp>(nl, av, prod);
          mlir::Value ins  = nb.create<mlir::tensor::InsertOp>(nl, out, acc, mlir::ValueRange{iv});
          nb.create<mlir::scf::YieldOp>(nl, ins);
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return mlir::success();
  }
};
} // namespace

namespace custom {

class CustomToLoops : public impl::CustomToLoopsBase<CustomToLoops> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                           mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
    target.addIllegalDialect<custom::CustomDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAddCMul>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createCustomToLoops() {
  return std::make_unique<CustomToLoops>();
}
} // namespace custom