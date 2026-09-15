add_mlir_dialect(customOps custom)

set(LLVM_TARGET_DEFINITIONS customPasses.td)
mlir_tablegen(customPasses.h.inc -gen-pass-decls -name Custom)
add_public_tablegen_target(MLIRcustomPassesIncGen)