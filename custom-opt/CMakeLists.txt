
add_llvm_executable(custom-opt custom-opt.cpp)

llvm_update_compile_flags(custom-opt)

target_link_libraries(custom-opt
    PRIVATE
    MLIRcustom
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRRegisterAllPasses
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTosaDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRAffineDialect
)

mlir_check_all_link_libraries(custom-opt)


