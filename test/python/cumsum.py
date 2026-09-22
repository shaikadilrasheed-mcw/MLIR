import os, sys, ctypes
import numpy as np

BUILD_DIR = "/home/mcw/Tharun/adil_custom_dialect/build"
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "custom"))

from mlir_custom.ir import Context, Module
from mlir_custom.passmanager import PassManager
from mlir_custom.execution_engine import ExecutionEngine
from mlir_custom.runtime import get_ranked_memref_descriptor
from mlir_custom.dialects import custom_nanobind as custom_d

with Context():
    custom_d.register_dialects()

    # cumsum: reduce a tensor<4xi64> (+ an i64 operand) down to an i64
    module = Module.parse(r"""
    func.func @cumsum(%a: tensor<4xi64>, %b: i64) -> i64
        attributes { llvm.emit_c_interface } {
      %0 = custom.cumsum %a, %b : (tensor<4xi64>, i64) -> i64
      return %0 : i64
    }
    """)

    # lower custom -> loops -> memref -> llvm
    pm = PassManager.parse(
        "builtin.module("
        "custom-to-loops,"
        "one-shot-bufferize{bufferize-function-boundaries},"
        "expand-strided-metadata,"
        "convert-scf-to-cf,"
        "convert-arith-to-llvm,"
        "finalize-memref-to-llvm,"
        "convert-cf-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)

    ee = ExecutionEngine(module)

    a   = np.array([10, 20, 30, 40], dtype=np.int64)   # 4 elements -> tensor<4xi64>
    b   = ctypes.c_int64(0)                            # the i64 operand
    res = ctypes.c_int64(-1)                           # result cell

    # memref arg -> pointer(pointer(descriptor)); scalars -> pointer; result LAST
    a_arg = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))
    ee.invoke("cumsum", a_arg, ctypes.pointer(b), ctypes.pointer(res))

    print(f"cumsum({a.tolist()}, {b.value}) = {res.value}")
    assert res.value == int(a.sum()), f"expected {int(a.sum())}, got {res.value}"
    print("OK")