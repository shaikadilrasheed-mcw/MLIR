include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(CustomPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CustomPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_custom"
  TD_FILE dialects/customOps.td
  SOURCES
    dialects/custom_nanobind.py
  DIALECT_NAME custom)

declare_mlir_python_extension(CustomPythonSources.NanobindExtension
  MODULE_NAME _customDialectsNanobind
  ADD_TO_PARENT CustomPythonSources
  SOURCES
    CustomExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPIMath
    MLIRCAPIMemRef
    MLIRCAPISCF
    MLIRCAPIControlFlow
    MLIRCAPILinalg
    MLIRCAPIBufferization
    MLIRCAPITensor
    MLIRCAPILLVM
    MLIRCAPITransforms
    CustomCAPI
)

add_mlir_python_common_capi_library(CustomPythonCAPI
  INSTALL_COMPONENT CustomPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    CustomPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.ExecutionEngine
)

add_mlir_python_modules(CustomPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    CustomPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    CustomPythonCAPI
)
