# File   : test_addcdiv.py
# Purpose: Implemented with Tosa, Test and compare the result of custom  addcdiv op and  addcdiv achieved with torch ops.
# Author : Adil
# Date   : 2026-09-24

import os, sys, ctypes
import numpy as np

BUILD_DIR = "/home/mcw/Tharun/adil_custom_dialect/build"
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "custom"))

from mlir_custom.ir import Context, Module
from mlir_custom.passmanager import PassManager
from mlir_custom.execution_engine import ExecutionEngine
from mlir_custom.dialects import custom_nanobind as custom_d
from mlir_custom.runtime import (get_ranked_memref_descriptor,
                                  make_nd_memref_descriptor,
                                  ranked_memref_to_numpy)

import torch


def memref(arr):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arr)))


# ======================= SCALAR (i64) =======================
def test_scalar():
    A, B, C = 10, 20, 5
    with Context():
        custom_d.register_dialects()
        module = Module.parse(r"""
        func.func @sc(%a: i64, %b: i64, %c: i64) -> i64
            attributes { llvm.emit_c_interface } {
          %r = custom.addcdiv %a, %b, %c : i64
          return %r : i64
        }
        """)
        PassManager.parse(
            "builtin.module("
            "custom-to-tosa,"
            "convert-arith-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts)"
        ).run(module.operation)

        ee = ExecutionEngine(module)
        a, b, c = ctypes.c_int64(A), ctypes.c_int64(B), ctypes.c_int64(C)
        res = ctypes.c_int64(-1)
        ee.invoke("sc", ctypes.pointer(a), ctypes.pointer(b),
                  ctypes.pointer(c), ctypes.pointer(res))

        divResult = torch.div(torch.tensor(B), torch.tensor(C)).item()
        result = torch.add(torch.tensor(A), divResult).item()
        print(f"[scalar] custom={res.value}  torch={result}")
        assert res.value == result
        print("[scalar] MATCH ✓")


# ======================= TENSOR (i32, via TOSA) =======================
def test_tensor():
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    b = np.array([ 10,  20,  30,  40], dtype=np.int32)
    c = np.array([ 5,  10,  15,  20], dtype=np.int32)

    with Context():
        custom_d.register_dialects()
        module = Module.parse(r"""
        func.func @tn(%a: tensor<4xi32>, %b: tensor<4xi32>, %c: tensor<4xi32>)
            -> tensor<4xi32> attributes { llvm.emit_c_interface } {
          %r = custom.addcdiv %a, %b, %c : tensor<4xi32>
          return %r : tensor<4xi32>
        }
        """)
        PassManager.parse(
            "builtin.module("
            "custom-to-tosa,"
            "func.func(tosa-to-linalg, tosa-to-arith),"
            "one-shot-bufferize{bufferize-function-boundaries},"
            "func.func(convert-linalg-to-loops),"
            "expand-strided-metadata,"
            "convert-scf-to-cf,"
            "convert-arith-to-llvm,"
            "finalize-memref-to-llvm,"
            "convert-cf-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts)"
        ).run(module.operation)

        ee = ExecutionEngine(module)

        # empty result descriptor (rank 1, i32) — the function fills it
        res_desc = make_nd_memref_descriptor(1, ctypes.c_int32)()
        res_ptr  = ctypes.pointer(ctypes.pointer(res_desc))

        ee.invoke("tn", res_ptr, memref(a), memref(b), memref(c))   # result FIRST
        custom = ranked_memref_to_numpy(ctypes.pointer(res_desc))   # read it back

        divResult = torch.div(torch.from_numpy(b), torch.from_numpy(c))
        result = torch.add(torch.from_numpy(a), divResult).numpy()  
        print(f"[tensor] custom={custom.tolist()} torch={result.tolist()}")
        assert np.array_equal(custom, result)
        print("[tensor] MATCH ✓")


if __name__ == "__main__":
    test_scalar()
    test_tensor()