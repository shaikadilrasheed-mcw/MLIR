func.func @test_add(%arg0 : i64, %arg1 : i64) -> i64 {
    %result = custom.add %arg0, %arg1 : i64
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

func.func @assignment() -> i64  {
	%a = custom.constant 10 : i64
    %b = custom.constant 4 : i64
    %c = custom.constant 5 : i64
    %d = custom.constant 30 : i64
    %mul_res = custom.mul %b, %c : i64
    %add_res = custom.add %a, %mul_res : i64
    %result = custom.sub %add_res, %d : i64
    return %result : i64
}
