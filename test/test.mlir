func.func @test_add(%arg0 : i64, %arg1 : i64) -> i64 {
    %result = custom.add %arg0, %arg1 : i64
    return %result : i64
}
