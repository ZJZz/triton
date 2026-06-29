// -----// IR Dump Before TritonGPUPipeline: tritongpu-pipeline{dump-intermediate-steps=true num-stages=3} ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":6:1)
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#loc25 = loc("a_ptr"(#loc))
#loc26 = loc("b_ptr"(#loc))
#loc27 = loc("c_ptr"(#loc))
#loc28 = loc("M"(#loc))
#loc29 = loc("N"(#loc))
#loc30 = loc("K"(#loc))
#loc31 = loc("stride_am"(#loc))
#loc32 = loc("stride_bk"(#loc))
#loc33 = loc("stride_cm"(#loc))
#loc34 = loc("stride_cn"(#loc))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_contiguous_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %M: i32 loc("M"(#loc)), %N: i32 loc("N"(#loc)), %K: i32 loc("K"(#loc)), %stride_am: i32 loc("stride_am"(#loc)), %stride_bk: i32 loc("stride_bk"(#loc)), %stride_cm: i32 loc("stride_cm"(#loc)), %stride_cn: i32 loc("stride_cn"(#loc))) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<64x32xi32, #blocked> loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc1)
    %pid_m = tt.get_program_id x : i32 loc(#loc35)
    %pid_n = tt.get_program_id y : i32 loc(#loc36)
    %offs_m = arith.muli %pid_m, %c64_i32 : i32 loc(#loc37)
    %offs_m_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc38)
    %offs_m_2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc38)
    %offs_m_3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc38)
    %offs_m_4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc38)
    %offs_m_5 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc37)
    %offs_m_6 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc37)
    %offs_m_7 = arith.addi %offs_m_5, %offs_m_1 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc37)
    %offs_m_8 = arith.addi %offs_m_6, %offs_m_2 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc37)
    %offs_n = arith.muli %pid_n, %c64_i32 : i32 loc(#loc39)
    %offs_n_9 = tt.splat %offs_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc39)
    %offs_n_10 = tt.splat %offs_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc39)
    %offs_n_11 = arith.addi %offs_n_9, %offs_m_3 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc39)
    %offs_n_12 = arith.addi %offs_n_10, %offs_m_4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc39)
    %a_ptrs = tt.expand_dims %offs_m_7 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc40)
    %a_ptrs_13 = tt.expand_dims %offs_m_8 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1> loc(#loc40)
    %a_ptrs_14 = tt.splat %stride_am : i32 -> tensor<64x1xi32, #blocked> loc(#loc40)
    %a_ptrs_15 = arith.muli %a_ptrs, %a_ptrs_14 : tensor<64x1xi32, #blocked> loc(#loc40)
    %a_ptrs_16 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked> loc(#loc41)
    %a_ptrs_17 = tt.addptr %a_ptrs_16, %a_ptrs_15 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked> loc(#loc41)
    %a_ptrs_18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc42)
    %a_ptrs_19 = tt.expand_dims %a_ptrs_18 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc42)
    %a_ptrs_20 = tt.broadcast %a_ptrs_17 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc41)
    %a_ptrs_21 = tt.broadcast %a_ptrs_19 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc41)
    %a_ptrs_22 = tt.addptr %a_ptrs_20, %a_ptrs_21 {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<[1, 32]> : tensor<2xi32>} : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc41)
    %b_ptrs = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc43)
    %b_ptrs_23 = tt.expand_dims %b_ptrs {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2> loc(#loc43)
    %b_ptrs_24 = tt.splat %stride_bk : i32 -> tensor<32x1xi32, #blocked2> loc(#loc43)
    %b_ptrs_25 = arith.muli %b_ptrs_23, %b_ptrs_24 : tensor<32x1xi32, #blocked2> loc(#loc43)
    %b_ptrs_26 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked2> loc(#loc44)
    %b_ptrs_27 = tt.addptr %b_ptrs_26, %b_ptrs_25 : tensor<32x1x!tt.ptr<f16>, #blocked2>, tensor<32x1xi32, #blocked2> loc(#loc44)
    %b_ptrs_28 = tt.expand_dims %offs_n_11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2> loc(#loc45)
    %b_ptrs_29 = tt.expand_dims %offs_n_12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc45)
    %b_ptrs_30 = tt.broadcast %b_ptrs_27 : tensor<32x1x!tt.ptr<f16>, #blocked2> -> tensor<32x64x!tt.ptr<f16>, #blocked2> loc(#loc44)
    %b_ptrs_31 = tt.broadcast %b_ptrs_28 : tensor<1x64xi32, #blocked2> -> tensor<32x64xi32, #blocked2> loc(#loc44)
    %b_ptrs_32 = tt.addptr %b_ptrs_30, %b_ptrs_31 {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<[1, 64]> : tensor<2xi32>} : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<32x64xi32, #blocked2> loc(#loc44)
    %b_ptrs_33 = arith.muli %stride_bk, %c32_i32 : i32 loc(#loc46)
    %b_ptrs_34 = tt.splat %b_ptrs_33 : i32 -> tensor<32x64xi32, #blocked2> loc(#loc47)
    %acc:3 = scf.for %_ = %c0_i32 to %K step %c32_i32 iter_args(%acc_43 = %cst_0, %a_ptrs_44 = %a_ptrs_22, %b_ptrs_45 = %b_ptrs_32) -> (tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked2>)  : i32 {
      %a = tt.load %a_ptrs_44 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc49)
      %b = tt.load %b_ptrs_45 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked2> loc(#loc50)
      %a_46 = ttg.convert_layout %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> loc(#loc49)
      %b_47 = ttg.convert_layout %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x64xf16, #blocked2> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> loc(#loc50)
      %acc_48 = tt.dot %a_46, %b_47, %acc_43, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma> loc(#loc51)
      %a_ptrs_49 = tt.addptr %a_ptrs_44, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc52)
      %b_ptrs_50 = tt.addptr %b_ptrs_45, %b_ptrs_34 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<32x64xi32, #blocked2> loc(#loc47)
      scf.yield %acc_48, %a_ptrs_49, %b_ptrs_50 : tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked2> loc(#loc2)
    } {tt.scheduled_max_stage = 2 : i32} loc(#loc58)
    %c = arith.truncf %acc#0 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma> loc(#loc53)
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked1> loc(#loc54)
    %c_ptrs_35 = arith.muli %c_ptrs, %a_ptrs_13 : tensor<64x1xi32, #blocked1> loc(#loc54)
    %c_ptrs_36 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1> loc(#loc55)
    %c_ptrs_37 = tt.addptr %c_ptrs_36, %c_ptrs_35 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1> loc(#loc55)
    %c_ptrs_38 = tt.splat %stride_cn : i32 -> tensor<1x64xi32, #blocked1> loc(#loc56)
    %c_ptrs_39 = arith.muli %c_ptrs_38, %b_ptrs_29 : tensor<1x64xi32, #blocked1> loc(#loc56)
    %c_ptrs_40 = tt.broadcast %c_ptrs_37 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x64x!tt.ptr<f16>, #blocked1> loc(#loc55)
    %c_ptrs_41 = tt.broadcast %c_ptrs_39 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1> loc(#loc55)
    %c_ptrs_42 = tt.addptr %c_ptrs_40, %c_ptrs_41 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1> loc(#loc55)
    %0 = ttg.convert_layout %c : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #blocked1> loc(#loc24)
    tt.store %c_ptrs_42, %0 : tensor<64x64x!tt.ptr<f16>, #blocked1> loc(#loc24)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":29:5)
#loc3 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":14:13)
#loc4 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":15:13)
#loc5 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":17:14)
#loc6 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":17:32)
#loc7 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":18:14)
#loc8 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":23:22)
#loc9 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":23:14)
#loc10 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":23:52)
#loc11 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":24:22)
#loc12 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":24:14)
#loc13 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":24:52)
#loc14 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":34:19)
#loc15 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":34:9)
#loc16 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":30:13)
#loc17 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":31:13)
#loc18 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":32:16)
#loc19 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":33:9)
#loc20 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":36:9)
#loc21 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":39:22)
#loc22 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":39:14)
#loc23 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":39:53)
#loc24 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py":40:5)
#loc35 = loc("pid_m"(#loc3))
#loc36 = loc("pid_n"(#loc4))
#loc37 = loc("offs_m"(#loc5))
#loc38 = loc("offs_m"(#loc6))
#loc39 = loc("offs_n"(#loc7))
#loc40 = loc("a_ptrs"(#loc8))
#loc41 = loc("a_ptrs"(#loc9))
#loc42 = loc("a_ptrs"(#loc10))
#loc43 = loc("b_ptrs"(#loc11))
#loc44 = loc("b_ptrs"(#loc12))
#loc45 = loc("b_ptrs"(#loc13))
#loc46 = loc("b_ptrs"(#loc14))
#loc47 = loc("b_ptrs"(#loc15))
#loc48 = loc("a_ptrs"(#loc2))
#loc49 = loc("a"(#loc16))
#loc50 = loc("b"(#loc17))
#loc51 = loc("acc"(#loc18))
#loc52 = loc("a_ptrs"(#loc19))
#loc53 = loc("c"(#loc20))
#loc54 = loc("c_ptrs"(#loc21))
#loc55 = loc("c_ptrs"(#loc22))
#loc56 = loc("c_ptrs"(#loc23))
#loc57 = loc("b_ptrs"(#loc48))
#loc58 = loc("acc"(#loc57))


[triton-loop-pipeline]: Load %52 = tt.load %arg12 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> has width 128
[triton-loop-pipeline]: Load %53 = tt.load %arg13 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> has width 128
// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_contiguous_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %7 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %8 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %9 = arith.addi %7, %3 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %10 = arith.addi %8, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %11 = arith.muli %1, %c64_i32 : i32
    %12 = tt.splat %11 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %13 = tt.splat %11 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %14 = arith.addi %12, %5 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %15 = arith.addi %13, %6 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %16 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %17 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %18 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %19 = arith.muli %16, %18 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %20 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %21 = tt.addptr %20, %19 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %24 = tt.broadcast %21 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.broadcast %23 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %26 = tt.addptr %24, %25 {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<[1, 32]> : tensor<2xi32>} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %27 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %28 = tt.expand_dims %27 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %29 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %30 = arith.muli %28, %29 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %31 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %32 = tt.addptr %31, %30 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %33 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %34 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %35 = tt.broadcast %32 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %36 = tt.broadcast %33 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %37 = tt.addptr %35, %36 {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<[1, 64]> : tensor<2xi32>} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %38 = arith.muli %arg7, %c32_i32 : i32
    %39 = tt.splat %38 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %40 = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %41 = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %42:5 = scf.for %arg10 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg11 = %cst_0, %arg12 = %26, %arg13 = %37, %arg14 = %c-1_i32, %arg15 = %c-1_i32) -> (tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32, i32)  : i32 {
      %c2_i32 = arith.constant {loop.cluster = 3 : i32, loop.stage = 0 : i32} 2 : i32
      %55 = arith.addi %arg14, %c1_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %56 = arith.cmpi sge, %55, %c2_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %57 = arith.select %56, %c0_i32_1, %55 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %58 = arith.addi %arg15, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %59 = arith.cmpi sge, %58, %c2_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %60 = arith.select %59, %c0_i32_1, %58 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %61 = ttg.memdesc_index %40[%57] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %62 = ttg.async_copy_global_to_local %arg12, %61 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %63 = ttg.async_commit_group tokens %62 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
      %64 = ttg.async_wait %63 {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
      %65 = ttg.memdesc_index %40[%60] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %66 = ttg.local_load %65 token %64 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %67 = ttg.memdesc_index %41[%57] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %68 = ttg.async_copy_global_to_local %arg13, %67 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %69 = ttg.async_commit_group tokens %68 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
      %70 = ttg.async_wait %69 {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
      %71 = ttg.memdesc_index %41[%60] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %72 = ttg.local_load %71 token %70 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %73 = ttg.convert_layout %66 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>>
      %74 = ttg.convert_layout %72 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>>
      %75 = tt.dot %73, %74, %arg11, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>> -> tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>
      %76 = tt.addptr %arg12, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %77 = tt.addptr %arg13, %39 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      scf.yield %75, %76, %77, %57, %60 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32, i32
    } {tt.scheduled_max_stage = 2 : i32}
    %43 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %41 : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %40 : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %44 = arith.truncf %42#0 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>> to tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>
    %45 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %46 = arith.muli %45, %17 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %47 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %48 = tt.addptr %47, %46 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %49 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %50 = arith.muli %49, %34 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %51 = tt.broadcast %48 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %52 = tt.broadcast %50 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %53 = tt.addptr %51, %52 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %54 = ttg.convert_layout %44 : tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.store %53, %54 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_contiguous_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {noinline = false} {
    %c2_i32 = arith.constant {loop.cluster = 3 : i32, loop.stage = 0 : i32} 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<32> : tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %7 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %8 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %9 = arith.addi %7, %3 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %10 = arith.addi %8, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %11 = arith.muli %1, %c64_i32 : i32
    %12 = tt.splat %11 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %13 = tt.splat %11 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %14 = arith.addi %12, %5 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %15 = arith.addi %13, %6 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %16 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %17 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %18 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %19 = arith.muli %16, %18 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %20 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %21 = tt.addptr %20, %19 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %24 = tt.broadcast %21 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.broadcast %23 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %26 = tt.addptr %24, %25 {tt.contiguity = dense<[1, 32]> : tensor<2xi32>, tt.divisibility = dense<[1, 32]> : tensor<2xi32>} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %27 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %28 = tt.expand_dims %27 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %29 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %30 = arith.muli %28, %29 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %31 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %32 = tt.addptr %31, %30 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %33 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %34 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %35 = tt.broadcast %32 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %36 = tt.broadcast %33 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %37 = tt.addptr %35, %36 {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<[1, 64]> : tensor<2xi32>} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %38 = arith.muli %arg7, %c32_i32 : i32
    %39 = tt.splat %38 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %40 = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %41 = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %42 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %43 = arith.cmpi sge, %c0_i32, %c2_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
    %44 = arith.select %43, %c0_i32, %c0_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
    %45 = ttg.memdesc_index %40[%44] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %46 = tt.splat %42 : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %47 = ttg.async_copy_global_to_local %26, %45 mask %46 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %48 = ttg.async_commit_group tokens %47 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %49 = ttg.memdesc_index %41[%44] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %50 = tt.splat %42 : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %51 = ttg.async_copy_global_to_local %37, %49 mask %50 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %52 = ttg.async_commit_group tokens %51 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %53 = arith.cmpi sgt, %arg5, %c32_i32 : i32
    %54 = tt.addptr %26, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %55 = tt.addptr %37, %39 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %56 = arith.addi %44, %c1_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
    %57 = arith.cmpi sge, %56, %c2_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
    %58 = arith.select %57, %c0_i32, %56 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
    %59 = ttg.memdesc_index %40[%58] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %60 = tt.splat %53 : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %61 = ttg.async_copy_global_to_local %54, %59 mask %60 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %62 = ttg.async_commit_group tokens %61 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %63 = ttg.memdesc_index %41[%58] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %64 = tt.splat %53 : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %65 = ttg.async_copy_global_to_local %55, %63 mask %64 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %66 = ttg.async_commit_group tokens %65 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
    %67:11 = scf.for %arg10 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg11 = %cst_0, %arg12 = %54, %arg13 = %55, %arg14 = %58, %arg15 = %c-1_i32, %arg16 = %c2_i32, %arg17 = %c2_i32, %arg18 = %48, %arg19 = %62, %arg20 = %52, %arg21 = %66) -> (tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %80 = arith.subi %arg5, %c64_i32 : i32
      %81 = arith.cmpi slt, %arg10, %80 : i32
      %82 = arith.subi %arg5, %c32_i32 : i32
      %83 = arith.cmpi slt, %arg10, %82 : i32
      %84 = arith.addi %arg15, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %85 = arith.cmpi sge, %84, %arg16 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %86 = arith.select %85, %c0_i32, %84 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %87 = ttg.async_wait %arg18 {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
      %88 = ttg.memdesc_index %40[%86] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %89 = ttg.local_load %88 token %87 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %90 = ttg.async_wait %arg20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
      %91 = ttg.memdesc_index %41[%86] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %92 = ttg.local_load %91 token %90 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %93 = ttg.convert_layout %89 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>>
      %94 = ttg.convert_layout %92 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>>
      %95 = tt.dot %93, %94, %arg11, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>, kWidth = 2}>> -> tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>
      %96 = tt.addptr %arg12, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %97 = tt.addptr %arg13, %39 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %98 = arith.addi %arg14, %c1_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %99 = arith.cmpi sge, %98, %c2_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %100 = arith.select %99, %c0_i32, %98 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %101 = ttg.memdesc_index %40[%100] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %102 = tt.splat %81 : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %103 = ttg.async_copy_global_to_local %96, %101 mask %102 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %104 = ttg.async_commit_group tokens %103 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
      %105 = ttg.memdesc_index %41[%100] {loop.cluster = 3 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %106 = tt.splat %81 : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %107 = ttg.async_copy_global_to_local %97, %105 mask %106 {contiguity = 8 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %108 = ttg.async_commit_group tokens %107 {loop.cluster = 3 : i32, loop.stage = 0 : i32}
      scf.yield %95, %96, %97, %100, %86, %arg17, %c2_i32, %arg19, %104, %arg21, %108 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32}
    %68 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %41 : !ttg.memdesc<2x32x64xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %40 : !ttg.memdesc<2x64x32xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %69 = arith.truncf %67#0 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>> to tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>>
    %70 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %71 = arith.muli %70, %17 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %72 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %73 = tt.addptr %72, %71 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %74 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %75 = arith.muli %74, %34 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %76 = tt.broadcast %73 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %77 = tt.broadcast %75 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %78 = tt.addptr %76, %77 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %79 = ttg.convert_layout %69 : tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.store %78, %79 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.return
  }
}


