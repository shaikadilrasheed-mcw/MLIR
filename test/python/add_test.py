import os, sys, ctypes

BUILD_DIR = "/home/mcw/Tharun/adil_custom_dialect/build"
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "custom"))

from mlir_custom.ir import Context, Module
from mlir_custom.passmanager import PassManager
from mlir_custom.execution_engine import ExecutionEngine
from mlir_custom.dialects import custom_nanobind as custom_d 

with Context():
    custom_d.register_dialects()

     # build IR that uses custom.add
    module = Module.parse(r"""
    func.func @add_i32(%a: i32, %b: i32) -> i32 attributes { llvm.emit_c_interface } {
          %0 = custom.add %a, %b : i32
          return %0 : i32
        }

    func.func @mul_i32(%a: i32, %b: i32) -> i32 attributes { llvm.emit_c_interface } {
      %0 = custom.mul %a, %b : i32
      return %0 : i32
    }
    """)

    # lower custom -> arith -> llvm
    pm = PassManager.parse(
        "builtin.module("
        "custom-to-arith,"
        "convert-arith-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)

    # JIT compile and run
    ee = ExecutionEngine(module)

    c_int = ctypes.c_int32 * 1
    a   = c_int(10)
    b   = c_int(20)
    res = c_int(-1)
    ee.invoke("add_i32", a, b, res)

    print(f"add({a[0]}, {b[0]}) = {res[0]}")
    assert res[0] == 30

    c_int = ctypes.c_int32 * 1
    a   = c_int(10)
    b   = c_int(20)
    res = c_int(-1)
    ee.invoke("mul_i32", a, b, res)

    print(f"mul({a[0]}, {b[0]}) = {res[0]}")
    assert res[0] == 200