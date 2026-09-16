// RUN: custom-opt %s --custom-to-tosa | FileCheck %s

//CHECK-LABEL: func.func @add_vector
func.func @add_vector(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  //CHECK-NEXT: %[[result:.*]] = tosa.add
  %0 = custom.add %a, %b : tensor<4xi32>
  //CHECK-NEXT: return  %[[result]]
  return %0 : tensor<4xi32>
}

func.func @sub_vector(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = custom.sub %a, %b : tensor<4xi32>
  return %0 : tensor<4xi32>
}
