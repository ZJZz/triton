// -----// IR Dump Before TritonGPUPipeline: tritongpu-pipeline{dump-intermediate-steps=true num-stages=3} ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#loc = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":6:1)
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#loc26 = loc("a_ptr"(#loc))
#loc27 = loc("b_ptr"(#loc))
#loc28 = loc("c_ptr"(#loc))
#loc29 = loc("M"(#loc))
#loc30 = loc("N"(#loc))
#loc31 = loc("K"(#loc))
#loc32 = loc("stride_am"(#loc))
#loc33 = loc("stride_ak"(#loc))
#loc34 = loc("stride_bk"(#loc))
#loc35 = loc("stride_bn"(#loc))
#loc36 = loc("stride_cm"(#loc))
#loc37 = loc("stride_cn"(#loc))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %M: i32 loc("M"(#loc)), %N: i32 loc("N"(#loc)), %K: i32 loc("K"(#loc)), %stride_am: i32 loc("stride_am"(#loc)), %stride_ak: i32 loc("stride_ak"(#loc)), %stride_bk: i32 loc("stride_bk"(#loc)), %stride_bn: i32 loc("stride_bn"(#loc)), %stride_cm: i32 loc("stride_cm"(#loc)), %stride_cn: i32 loc("stride_cn"(#loc))) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc1)
    %pid_m = tt.get_program_id x : i32 loc(#loc38)
    %pid_n = tt.get_program_id y : i32 loc(#loc39)
    %offs_m = arith.muli %pid_m, %c64_i32 : i32 loc(#loc40)
    %offs_m_0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc41)
    %offs_m_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc41)
    %offs_m_2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc41)
    %offs_m_3 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc40)
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc40)
    %offs_m_5 = arith.addi %offs_m_3, %offs_m_0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc40)
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_1 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc40)
    %offs_n = arith.muli %pid_n, %c64_i32 : i32 loc(#loc42)
    %offs_n_7 = tt.splat %offs_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc42)
    %offs_n_8 = arith.addi %offs_n_7, %offs_m_2 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc42)
    %a_ptrs = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc43)
    %a_ptrs_9 = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1> loc(#loc43)
    %a_ptrs_10 = tt.splat %stride_am : i32 -> tensor<64x1xi32, #blocked> loc(#loc43)
    %a_ptrs_11 = arith.muli %a_ptrs, %a_ptrs_10 : tensor<64x1xi32, #blocked> loc(#loc43)
    %a_ptrs_12 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked> loc(#loc44)
    %a_ptrs_13 = tt.addptr %a_ptrs_12, %a_ptrs_11 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked> loc(#loc44)
    %a_ptrs_14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc45)
    %a_ptrs_15 = tt.expand_dims %a_ptrs_14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc45)
    %a_ptrs_16 = tt.splat %stride_ak : i32 -> tensor<1x32xi32, #blocked> loc(#loc45)
    %a_ptrs_17 = arith.muli %a_ptrs_15, %a_ptrs_16 : tensor<1x32xi32, #blocked> loc(#loc45)
    %a_ptrs_18 = tt.broadcast %a_ptrs_13 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc44)
    %a_ptrs_19 = tt.broadcast %a_ptrs_17 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc44)
    %a_ptrs_20 = tt.addptr %a_ptrs_18, %a_ptrs_19 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc44)
    %b_ptrs = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc46)
    %b_ptrs_21 = tt.expand_dims %b_ptrs {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1> loc(#loc46)
    %b_ptrs_22 = tt.splat %stride_bk : i32 -> tensor<32x1xi32, #blocked1> loc(#loc46)
    %b_ptrs_23 = arith.muli %b_ptrs_21, %b_ptrs_22 : tensor<32x1xi32, #blocked1> loc(#loc46)
    %b_ptrs_24 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked1> loc(#loc47)
    %b_ptrs_25 = tt.addptr %b_ptrs_24, %b_ptrs_23 : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi32, #blocked1> loc(#loc47)
    %b_ptrs_26 = tt.expand_dims %offs_n_8 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc48)
    %b_ptrs_27 = tt.splat %stride_bn : i32 -> tensor<1x64xi32, #blocked1> loc(#loc48)
    %b_ptrs_28 = arith.muli %b_ptrs_26, %b_ptrs_27 : tensor<1x64xi32, #blocked1> loc(#loc48)
    %b_ptrs_29 = tt.broadcast %b_ptrs_25 : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x64x!tt.ptr<f16>, #blocked1> loc(#loc47)
    %b_ptrs_30 = tt.broadcast %b_ptrs_28 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1> loc(#loc47)
    %b_ptrs_31 = tt.addptr %b_ptrs_29, %b_ptrs_30 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1> loc(#loc47)
    %a_ptrs_32 = arith.muli %stride_ak, %c32_i32 : i32 loc(#loc49)
    %a_ptrs_33 = tt.splat %a_ptrs_32 : i32 -> tensor<64x32xi32, #blocked> loc(#loc50)
    %b_ptrs_34 = arith.muli %stride_bk, %c32_i32 : i32 loc(#loc51)
    %b_ptrs_35 = tt.splat %b_ptrs_34 : i32 -> tensor<32x64xi32, #blocked1> loc(#loc52)
    %acc:3 = scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%a_ptrs_44 = %a_ptrs_20, %b_ptrs_45 = %b_ptrs_31, %acc_46 = %cst) -> (tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xf32, #mma>)  : i32 {
      %a = tt.load %a_ptrs_44 : tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc54)
      %a_47 = ttg.local_alloc %a : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem> loc(#loc54)
      %b = tt.load %b_ptrs_45 : tensor<32x64x!tt.ptr<f16>, #blocked1> loc(#loc55)
      %b_48 = ttg.local_alloc %b : (tensor<32x64xf16, #blocked1>) -> !ttg.memdesc<32x64xf16, #shared1, #smem> loc(#loc55)
      %acc_49 = ttng.warp_group_dot %a_47, %b_48, %acc_46 {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #shared, #smem> * !ttg.memdesc<32x64xf16, #shared1, #smem> -> tensor<64x64xf32, #mma> loc(#loc56)
      %a_ptrs_50 = tt.addptr %a_ptrs_44, %a_ptrs_33 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc50)
      %b_ptrs_51 = tt.addptr %b_ptrs_45, %b_ptrs_35 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1> loc(#loc52)
      scf.yield %a_ptrs_50, %b_ptrs_51, %acc_49 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xf32, #mma> loc(#loc2)
    } loc(#loc62)
    %c = arith.truncf %acc#2 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma> loc(#loc57)
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked1> loc(#loc58)
    %c_ptrs_36 = arith.muli %c_ptrs, %a_ptrs_9 : tensor<64x1xi32, #blocked1> loc(#loc58)
    %c_ptrs_37 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1> loc(#loc59)
    %c_ptrs_38 = tt.addptr %c_ptrs_37, %c_ptrs_36 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1> loc(#loc59)
    %c_ptrs_39 = tt.splat %stride_cn : i32 -> tensor<1x64xi32, #blocked1> loc(#loc60)
    %c_ptrs_40 = arith.muli %c_ptrs_39, %b_ptrs_26 : tensor<1x64xi32, #blocked1> loc(#loc60)
    %c_ptrs_41 = tt.broadcast %c_ptrs_38 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x64x!tt.ptr<f16>, #blocked1> loc(#loc59)
    %c_ptrs_42 = tt.broadcast %c_ptrs_40 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1> loc(#loc59)
    %c_ptrs_43 = tt.addptr %c_ptrs_41, %c_ptrs_42 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1> loc(#loc59)
    %0 = ttg.convert_layout %c : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #blocked1> loc(#loc25)
    tt.store %c_ptrs_43, %0 : tensor<64x64x!tt.ptr<f16>, #blocked1> loc(#loc25)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":26:5)
#loc3 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":15:13)
#loc4 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":16:13)
#loc5 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":18:14)
#loc6 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":18:32)
#loc7 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":19:14)
#loc8 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":22:22)
#loc9 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":22:14)
#loc10 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":22:52)
#loc11 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":23:22)
#loc12 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":23:14)
#loc13 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":23:52)
#loc14 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":30:19)
#loc15 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":30:9)
#loc16 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":31:19)
#loc17 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":31:9)
#loc18 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":27:13)
#loc19 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":28:13)
#loc20 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":29:16)
#loc21 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":33:9)
#loc22 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":36:22)
#loc23 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":36:14)
#loc24 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":36:53)
#loc25 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/matmul.py":37:5)
#loc38 = loc("pid_m"(#loc3))
#loc39 = loc("pid_n"(#loc4))
#loc40 = loc("offs_m"(#loc5))
#loc41 = loc("offs_m"(#loc6))
#loc42 = loc("offs_n"(#loc7))
#loc43 = loc("a_ptrs"(#loc8))
#loc44 = loc("a_ptrs"(#loc9))
#loc45 = loc("a_ptrs"(#loc10))
#loc46 = loc("b_ptrs"(#loc11))
#loc47 = loc("b_ptrs"(#loc12))
#loc48 = loc("b_ptrs"(#loc13))
#loc49 = loc("a_ptrs"(#loc14))
#loc50 = loc("a_ptrs"(#loc15))
#loc51 = loc("b_ptrs"(#loc16))
#loc52 = loc("b_ptrs"(#loc17))
#loc53 = loc("a_ptrs"(#loc2))
#loc54 = loc("a"(#loc18))
#loc55 = loc("b"(#loc19))
#loc56 = loc("acc"(#loc20))
#loc57 = loc("c"(#loc21))
#loc58 = loc("c_ptrs"(#loc22))
#loc59 = loc("c_ptrs"(#loc23))
#loc60 = loc("c_ptrs"(#loc24))
#loc61 = loc("b_ptrs"(#loc53))
#loc62 = loc("acc"(#loc61))


// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %6 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %7 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %8 = arith.addi %6, %3 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %9 = arith.addi %7, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %10 = arith.muli %1, %c64_i32 : i32
    %11 = tt.splat %10 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %12 = arith.addi %11, %5 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %13 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %14 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %15 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %16 = arith.muli %13, %15 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %17 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %18 = tt.addptr %17, %16 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %21 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %22 = arith.muli %20, %21 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %23 = tt.broadcast %18 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %24 = tt.broadcast %22 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.addptr %23, %24 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %27 = tt.expand_dims %26 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %28 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %29 = arith.muli %27, %28 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %30 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %31 = tt.addptr %30, %29 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %32 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %33 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %34 = arith.muli %32, %33 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %35 = tt.broadcast %31 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %36 = tt.broadcast %34 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %37 = tt.addptr %35, %36 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %38 = arith.muli %arg7, %c32_i32 : i32
    %39 = tt.splat %38 : i32 -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %40 = arith.muli %arg8, %c32_i32 : i32
    %41 = tt.splat %40 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %42:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg13 = %25, %arg14 = %37, %arg15 = %cst) -> (tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>)  : i32 {
      %54 = tt.load %arg13 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %55 = ttg.local_alloc %54 : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %56 = tt.load %arg14 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %57 = ttg.local_alloc %56 : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %58 = ttng.warp_group_dot %55, %57, %arg15 {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
      %59 = tt.addptr %arg13, %39 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %60 = tt.addptr %arg14, %41 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      scf.yield %59, %60, %58 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
    }
    %43 = arith.truncf %42#2 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>> to tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
    %44 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %45 = arith.muli %44, %14 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %46 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %47 = tt.addptr %46, %45 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %48 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %49 = arith.muli %48, %32 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %50 = tt.broadcast %47 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %51 = tt.broadcast %49 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %52 = tt.addptr %50, %51 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %53 = ttg.convert_layout %43 : tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.store %52, %53 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %6 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %7 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %8 = arith.addi %6, %3 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %9 = arith.addi %7, %4 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %10 = arith.muli %1, %c64_i32 : i32
    %11 = tt.splat %10 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %12 = arith.addi %11, %5 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %13 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %14 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %15 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %16 = arith.muli %13, %15 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %17 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %18 = tt.addptr %17, %16 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %21 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %22 = arith.muli %20, %21 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %23 = tt.broadcast %18 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %24 = tt.broadcast %22 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.addptr %23, %24 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %27 = tt.expand_dims %26 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %28 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %29 = arith.muli %27, %28 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %30 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %31 = tt.addptr %30, %29 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %32 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %33 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %34 = arith.muli %32, %33 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %35 = tt.broadcast %31 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %36 = tt.broadcast %34 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %37 = tt.addptr %35, %36 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %38 = arith.muli %arg7, %c32_i32 : i32
    %39 = tt.splat %38 : i32 -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %40 = arith.muli %arg8, %c32_i32 : i32
    %41 = tt.splat %40 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %42:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg13 = %25, %arg14 = %37, %arg15 = %cst) -> (tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>)  : i32 {
      %54 = tt.load %arg13 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %55 = ttg.local_alloc %54 : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %56 = tt.load %arg14 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %57 = ttg.local_alloc %56 : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %58 = ttng.warp_group_dot %55, %57, %arg15 {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> * !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
      %59 = tt.addptr %arg13, %39 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %60 = tt.addptr %arg14, %41 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      scf.yield %59, %60, %58 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
    }
    %43 = arith.truncf %42#2 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>> to tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>>
    %44 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %45 = arith.muli %44, %14 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %46 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %47 = tt.addptr %46, %45 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %48 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %49 = arith.muli %48, %32 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %50 = tt.broadcast %47 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %51 = tt.broadcast %49 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %52 = tt.addptr %50, %51 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %53 = ttg.convert_layout %43 : tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.store %52, %53 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.return
  }
}


