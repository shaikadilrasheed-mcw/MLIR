// RUN: custom-opt %s --custom-to-arith | FileCheck %s

//CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0 : i64, %arg1 : i64) -> i64 {
    //CHECK-NEXT: %[[result:.*]] = arith.add
    %result = custom.add %arg0, %arg1 : i64
    //CHECK-NEXT: return  %[[result]]
    return %result : i64
}

func.func @test_sub(%arg0 : i64, %arg1 : i64) -> i64 {
    %result = custom.sub %arg0, %arg1 : i64
    return %result : i64
}

func.func @test_mul(%arg0 : i64, %arg1 : i64) -> i64 {
    %result = custom.mul %arg0, %arg1 : i64
    return %result : i64
}

func.func @test_constant() -> i64  {
	%0 = custom.constant 40 : i64
    return %0 : i64
}

func.func @add_vector(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = custom.add %a, %b : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func.func @sub_vector(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = custom.sub %a, %b : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func.func @mul_vector(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = custom.mul %a, %b : tensor<4xi32>
  return %0 : tensor<4xi32>
}
