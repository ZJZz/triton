// -----// IR Dump Before TritonGPUPipeline: tritongpu-pipeline{dump-intermediate-steps=true num-stages=3} ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#loc = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":6:1)
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
#loc27 = loc("a_ptr"(#loc))
#loc28 = loc("b_ptr"(#loc))
#loc29 = loc("c_ptr"(#loc))
#loc30 = loc("M"(#loc))
#loc31 = loc("N"(#loc))
#loc32 = loc("K"(#loc))
#loc33 = loc("stride_am"(#loc))
#loc34 = loc("stride_ak"(#loc))
#loc35 = loc("stride_bk"(#loc))
#loc36 = loc("stride_bn"(#loc))
#loc37 = loc("stride_cm"(#loc))
#loc38 = loc("stride_cn"(#loc))
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %M: i32 loc("M"(#loc)), %N: i32 loc("N"(#loc)), %K: i32 loc("K"(#loc)), %stride_am: i32 loc("stride_am"(#loc)), %stride_ak: i32 loc("stride_ak"(#loc)), %stride_bk: i32 loc("stride_bk"(#loc)), %stride_bn: i32 loc("stride_bn"(#loc)), %stride_cm: i32 loc("stride_cm"(#loc)), %stride_cn: i32 loc("stride_cn"(#loc))) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma> loc(#loc1)
    %pid_m = tt.get_program_id x : i32 loc(#loc39)
    %pid_n = tt.get_program_id y : i32 loc(#loc40)
    %offs_m = arith.muli %pid_m, %c64_i32 : i32 loc(#loc41)
    %offs_m_0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc42)
    %offs_m_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc42)
    %offs_m_2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc42)
    %offs_m_3 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc41)
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc41)
    %offs_m_5 = arith.addi %offs_m_3, %offs_m_0 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc41)
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_2 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc41)
    %offs_n = arith.muli %pid_n, %c64_i32 : i32 loc(#loc43)
    %offs_n_7 = tt.splat %offs_n : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc43)
    %offs_n_8 = arith.addi %offs_n_7, %offs_m_1 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc43)
    %offs_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc44)
    %offs_k_9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc44)
    %a_ptrs = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc45)
    %a_ptrs_10 = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2> loc(#loc45)
    %a_ptrs_11 = tt.splat %stride_am : i32 -> tensor<64x1xi32, #blocked> loc(#loc45)
    %a_ptrs_12 = arith.muli %a_ptrs, %a_ptrs_11 : tensor<64x1xi32, #blocked> loc(#loc45)
    %a_ptrs_13 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked> loc(#loc46)
    %a_ptrs_14 = tt.addptr %a_ptrs_13, %a_ptrs_12 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked> loc(#loc46)
    %a_ptrs_15 = tt.expand_dims %offs_k_9 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked> loc(#loc47)
    %a_ptrs_16 = tt.splat %stride_ak : i32 -> tensor<1x32xi32, #blocked> loc(#loc47)
    %a_ptrs_17 = arith.muli %a_ptrs_15, %a_ptrs_16 : tensor<1x32xi32, #blocked> loc(#loc47)
    %a_ptrs_18 = tt.broadcast %a_ptrs_14 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc46)
    %a_ptrs_19 = tt.broadcast %a_ptrs_17 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked> loc(#loc46)
    %a_ptrs_20 = tt.addptr %a_ptrs_18, %a_ptrs_19 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc46)
    %b_ptrs = tt.expand_dims %offs_k {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2> loc(#loc48)
    %b_ptrs_21 = tt.splat %stride_bk : i32 -> tensor<32x1xi32, #blocked2> loc(#loc48)
    %b_ptrs_22 = arith.muli %b_ptrs, %b_ptrs_21 : tensor<32x1xi32, #blocked2> loc(#loc48)
    %b_ptrs_23 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked2> loc(#loc49)
    %b_ptrs_24 = tt.addptr %b_ptrs_23, %b_ptrs_22 : tensor<32x1x!tt.ptr<f16>, #blocked2>, tensor<32x1xi32, #blocked2> loc(#loc49)
    %b_ptrs_25 = tt.expand_dims %offs_n_8 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc50)
    %b_ptrs_26 = tt.splat %stride_bn : i32 -> tensor<1x64xi32, #blocked1> loc(#loc50)
    %b_ptrs_27 = arith.muli %b_ptrs_25, %b_ptrs_26 : tensor<1x64xi32, #blocked1> loc(#loc50)
    %b_ptrs_28 = tt.broadcast %b_ptrs_24 : tensor<32x1x!tt.ptr<f16>, #blocked2> -> tensor<32x64x!tt.ptr<f16>, #blocked2> loc(#loc49)
    %b_ptrs_29 = ttg.convert_layout %b_ptrs_27 : tensor<1x64xi32, #blocked1> -> tensor<1x64xi32, #blocked2> loc(#loc49)
    %b_ptrs_30 = tt.broadcast %b_ptrs_29 : tensor<1x64xi32, #blocked2> -> tensor<32x64xi32, #blocked2> loc(#loc49)
    %b_ptrs_31 = tt.addptr %b_ptrs_28, %b_ptrs_30 : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<32x64xi32, #blocked2> loc(#loc49)
    %a_ptrs_32 = arith.muli %stride_ak, %c32_i32 : i32 loc(#loc51)
    %a_ptrs_33 = tt.splat %a_ptrs_32 : i32 -> tensor<64x32xi32, #blocked> loc(#loc52)
    %b_ptrs_34 = arith.muli %stride_bk, %c32_i32 : i32 loc(#loc53)
    %b_ptrs_35 = tt.splat %b_ptrs_34 : i32 -> tensor<32x64xi32, #blocked2> loc(#loc54)
    %acc:3 = scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%b_ptrs_45 = %b_ptrs_31, %acc_46 = %cst, %a_ptrs_47 = %a_ptrs_20) -> (tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>)  : i32 {
      %a = tt.load %a_ptrs_47 : tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc56)
      %a_48 = ttg.local_alloc %a : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem> loc(#loc56)
      %b = tt.load %b_ptrs_45 : tensor<32x64x!tt.ptr<f16>, #blocked2> loc(#loc57)
      %b_49 = ttg.local_alloc %b : (tensor<32x64xf16, #blocked2>) -> !ttg.memdesc<32x64xf16, #shared1, #smem> loc(#loc57)
      %acc_50 = ttng.warp_group_dot %a_48, %b_49, %acc_46 {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #shared, #smem> * !ttg.memdesc<32x64xf16, #shared1, #smem> -> tensor<64x64xf32, #mma> loc(#loc58)
      %a_ptrs_51 = tt.addptr %a_ptrs_47, %a_ptrs_33 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc52)
      %b_ptrs_52 = tt.addptr %b_ptrs_45, %b_ptrs_35 : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<32x64xi32, #blocked2> loc(#loc54)
      scf.yield %b_ptrs_52, %acc_50, %a_ptrs_51 : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc2)
    } loc(#loc64)
    %c = arith.truncf %acc#1 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma> loc(#loc59)
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked2> loc(#loc60)
    %c_ptrs_36 = arith.muli %c_ptrs, %a_ptrs_10 : tensor<64x1xi32, #blocked2> loc(#loc60)
    %c_ptrs_37 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked2> loc(#loc61)
    %c_ptrs_38 = tt.addptr %c_ptrs_37, %c_ptrs_36 : tensor<64x1x!tt.ptr<f16>, #blocked2>, tensor<64x1xi32, #blocked2> loc(#loc61)
    %c_ptrs_39 = tt.splat %stride_cn : i32 -> tensor<1x64xi32, #blocked2> loc(#loc62)
    %c_ptrs_40 = ttg.convert_layout %b_ptrs_25 : tensor<1x64xi32, #blocked1> -> tensor<1x64xi32, #blocked2> loc(#loc62)
    %c_ptrs_41 = arith.muli %c_ptrs_39, %c_ptrs_40 : tensor<1x64xi32, #blocked2> loc(#loc62)
    %c_ptrs_42 = tt.broadcast %c_ptrs_38 : tensor<64x1x!tt.ptr<f16>, #blocked2> -> tensor<64x64x!tt.ptr<f16>, #blocked2> loc(#loc61)
    %c_ptrs_43 = tt.broadcast %c_ptrs_41 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2> loc(#loc61)
    %c_ptrs_44 = tt.addptr %c_ptrs_42, %c_ptrs_43 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2> loc(#loc61)
    %0 = ttg.convert_layout %c : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #blocked2> loc(#loc26)
    tt.store %c_ptrs_44, %0 : tensor<64x64x!tt.ptr<f16>, #blocked2> loc(#loc26)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":26:5)
#loc3 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":15:13)
#loc4 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":16:13)
#loc5 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":18:14)
#loc6 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":18:32)
#loc7 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":19:14)
#loc8 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":20:14)
#loc9 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:22)
#loc10 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:14)
#loc11 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:52)
#loc12 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:22)
#loc13 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:14)
#loc14 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:52)
#loc15 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":30:19)
#loc16 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":30:9)
#loc17 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":31:19)
#loc18 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":31:9)
#loc19 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":27:13)
#loc20 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":28:13)
#loc21 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":29:16)
#loc22 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":33:9)
#loc23 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:22)
#loc24 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:14)
#loc25 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:53)
#loc26 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":37:5)
#loc39 = loc("pid_m"(#loc3))
#loc40 = loc("pid_n"(#loc4))
#loc41 = loc("offs_m"(#loc5))
#loc42 = loc("offs_m"(#loc6))
#loc43 = loc("offs_n"(#loc7))
#loc44 = loc("offs_k"(#loc8))
#loc45 = loc("a_ptrs"(#loc9))
#loc46 = loc("a_ptrs"(#loc10))
#loc47 = loc("a_ptrs"(#loc11))
#loc48 = loc("b_ptrs"(#loc12))
#loc49 = loc("b_ptrs"(#loc13))
#loc50 = loc("b_ptrs"(#loc14))
#loc51 = loc("a_ptrs"(#loc15))
#loc52 = loc("a_ptrs"(#loc16))
#loc53 = loc("b_ptrs"(#loc17))
#loc54 = loc("b_ptrs"(#loc18))
#loc55 = loc("a_ptrs"(#loc2))
#loc56 = loc("a"(#loc19))
#loc57 = loc("b"(#loc20))
#loc58 = loc("acc"(#loc21))
#loc59 = loc("c"(#loc22))
#loc60 = loc("c_ptrs"(#loc23))
#loc61 = loc("c_ptrs"(#loc24))
#loc62 = loc("c_ptrs"(#loc25))
#loc63 = loc("b_ptrs"(#loc55))
#loc64 = loc("acc"(#loc63))


// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %6 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %7 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %8 = arith.addi %6, %3 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %9 = arith.addi %7, %5 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %10 = arith.muli %1, %c64_i32 : i32
    %11 = tt.splat %10 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %12 = arith.addi %11, %4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %15 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %16 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %17 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %18 = arith.muli %15, %17 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %19 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %20 = tt.addptr %19, %18 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %21 = tt.expand_dims %14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %22 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %23 = arith.muli %21, %22 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %24 = tt.broadcast %20 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %25 = tt.broadcast %23 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %26 = tt.addptr %24, %25 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %27 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %28 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %29 = arith.muli %27, %28 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %30 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %31 = tt.addptr %30, %29 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %32 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %33 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %34 = arith.muli %32, %33 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %35 = tt.broadcast %31 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %36 = ttg.convert_layout %34 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %37 = tt.broadcast %36 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %38 = tt.addptr %35, %37 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %39 = arith.muli %arg7, %c32_i32 : i32
    %40 = tt.splat %39 : i32 -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %41 = arith.muli %arg8, %c32_i32 : i32
    %42 = tt.splat %41 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %43:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg13 = %38, %arg14 = %cst, %arg15 = %26) -> (tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>)  : i32 {
      %56 = tt.load %arg15 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %57 = ttg.local_alloc %56 : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>
      %58 = tt.load %arg13 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %59 = ttg.local_alloc %58 : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      %60 = ttng.warp_group_dot %57, %59, %arg14 {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory> * !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory> -> tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>
      %61 = tt.addptr %arg15, %40 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %62 = tt.addptr %arg13, %42 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      scf.yield %62, %60, %61 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    }
    %44 = arith.truncf %43#1 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>> to tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>
    %45 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %46 = arith.muli %45, %16 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %47 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %48 = tt.addptr %47, %46 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %49 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %50 = ttg.convert_layout %32 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %51 = arith.muli %49, %50 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %52 = tt.broadcast %48 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %53 = tt.broadcast %51 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %54 = tt.addptr %52, %53 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %55 = ttg.convert_layout %44 : tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.store %54, %55 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %6 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %7 = tt.splat %2 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %8 = arith.addi %6, %3 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %9 = arith.addi %7, %5 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %10 = arith.muli %1, %c64_i32 : i32
    %11 = tt.splat %10 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %12 = arith.addi %11, %4 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %15 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %16 = tt.expand_dims %9 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %17 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %18 = arith.muli %15, %17 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %19 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %20 = tt.addptr %19, %18 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %21 = tt.expand_dims %14 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %22 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %23 = arith.muli %21, %22 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %24 = tt.broadcast %20 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %25 = tt.broadcast %23 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %26 = tt.addptr %24, %25 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %27 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %28 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %29 = arith.muli %27, %28 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %30 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %31 = tt.addptr %30, %29 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %32 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %33 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %34 = arith.muli %32, %33 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %35 = tt.broadcast %31 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %36 = ttg.convert_layout %34 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %37 = tt.broadcast %36 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %38 = tt.addptr %35, %37 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %39 = arith.muli %arg7, %c32_i32 : i32
    %40 = tt.splat %39 : i32 -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %41 = arith.muli %arg8, %c32_i32 : i32
    %42 = tt.splat %41 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %43:3 = scf.for %arg12 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg13 = %38, %arg14 = %cst, %arg15 = %26) -> (tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>)  : i32 {
      %56 = tt.load %arg15 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %57 = ttg.local_alloc %56 : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>
      %58 = tt.load %arg13 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %59 = ttg.local_alloc %58 : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      %60 = ttng.warp_group_dot %57, %59, %arg14 {inputPrecision = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory> * !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory> -> tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>
      %61 = tt.addptr %arg15, %40 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %62 = tt.addptr %arg13, %42 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      scf.yield %62, %60, %61 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    }
    %44 = arith.truncf %43#1 : tensor<64x64xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>> to tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>>
    %45 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %46 = arith.muli %45, %16 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %47 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %48 = tt.addptr %47, %46 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %49 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %50 = ttg.convert_layout %32 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %51 = arith.muli %49, %50 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %52 = tt.broadcast %48 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %53 = tt.broadcast %51 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %54 = tt.addptr %52, %53 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %55 = ttg.convert_layout %44 : tensor<64x64xf16, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[0, 1]], instrShape = [16, 32, 16]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.store %54, %55 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.return
  }
}


