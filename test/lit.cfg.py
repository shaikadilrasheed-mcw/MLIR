import os
import lit.formats

config.name = "custom"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# Put your custom-opt and LLVM's FileCheck on PATH so RUN lines find them
project_root = os.path.dirname(config.test_source_root)   # adil_custom_dialect/
custom_bin   = os.path.join(project_root, "build", "bin")  # your custom-opt lives here
llvm_bin     = "/home/mcw/Tharun/llvm-project/build/bin"    # FileCheck, llvm-lit, etc.

config.environment["PATH"] = os.pathsep.join(
    [custom_bin, llvm_bin, config.environment["PATH"]]
)