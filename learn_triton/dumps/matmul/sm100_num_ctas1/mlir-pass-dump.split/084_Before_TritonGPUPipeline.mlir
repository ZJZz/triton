// -----// IR Dump Before TritonGPUPipeline: tritongpu-pipeline{dump-intermediate-steps=true num-stages=3} ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>
#loc = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":6:1)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
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
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %M: i32 loc("M"(#loc)), %N: i32 loc("N"(#loc)), %K: i32 loc("K"(#loc)), %stride_am: i32 loc("stride_am"(#loc)), %stride_ak: i32 loc("stride_ak"(#loc)), %stride_bk: i32 loc("stride_bk"(#loc)), %stride_bn: i32 loc("stride_bn"(#loc)), %stride_cm: i32 loc("stride_cm"(#loc)), %stride_cn: i32 loc("stride_cn"(#loc))) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #linear> loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %false = arith.constant false loc(#loc1)
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
    %acc, %acc_36 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc53)
    %acc_37 = ttng.tmem_store %cst, %acc[%acc_36], %true : tensor<64x64xf32, #linear> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc53)
    %acc_38:4 = scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%a_ptrs_49 = %a_ptrs_20, %b_ptrs_50 = %b_ptrs_31, %acc_51 = %false, %acc_52 = %acc_37) -> (tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>, i1, !ttg.async.token)  : i32 {
      %a = tt.load %a_ptrs_49 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc55)
      %a_53 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem> loc(#loc55)
      %b = tt.load %b_ptrs_50 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked1> loc(#loc56)
      %b_54 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #blocked1>) -> !ttg.memdesc<32x64xf16, #shared1, #smem> loc(#loc56)
      %acc_55 = ttng.tc_gen5_mma %a_53, %b_54, %acc[%acc_52], %acc_51, %true {loop.cluster = 0 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<64x32xf16, #shared, #smem>, !ttg.memdesc<32x64xf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc53)
      %a_ptrs_56 = tt.addptr %a_ptrs_49, %a_ptrs_33 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc50)
      %b_ptrs_57 = tt.addptr %b_ptrs_50, %b_ptrs_35 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1> loc(#loc52)
      scf.yield %a_ptrs_56, %b_ptrs_57, %true, %acc_55 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>, i1, !ttg.async.token loc(#loc18)
    } {tt.scheduled_max_stage = 0 : i32} loc(#loc62)
    %acc_39, %acc_40 = ttng.tmem_load %acc[%acc_38#3] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear> loc(#loc53)
    %c = arith.truncf %acc_39 : tensor<64x64xf32, #linear> to tensor<64x64xf16, #linear> loc(#loc57)
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked1> loc(#loc58)
    %c_ptrs_41 = arith.muli %c_ptrs, %a_ptrs_9 : tensor<64x1xi32, #blocked1> loc(#loc58)
    %c_ptrs_42 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1> loc(#loc59)
    %c_ptrs_43 = tt.addptr %c_ptrs_42, %c_ptrs_41 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1> loc(#loc59)
    %c_ptrs_44 = tt.splat %stride_cn : i32 -> tensor<1x64xi32, #blocked1> loc(#loc60)
    %c_ptrs_45 = arith.muli %c_ptrs_44, %b_ptrs_26 : tensor<1x64xi32, #blocked1> loc(#loc60)
    %c_ptrs_46 = tt.broadcast %c_ptrs_43 : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x64x!tt.ptr<f16>, #blocked1> loc(#loc59)
    %c_ptrs_47 = tt.broadcast %c_ptrs_45 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1> loc(#loc59)
    %c_ptrs_48 = tt.addptr %c_ptrs_46, %c_ptrs_47 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1> loc(#loc59)
    %0 = ttg.convert_layout %c : tensor<64x64xf16, #linear> -> tensor<64x64xf16, #blocked1> loc(#loc25)
    tt.store %c_ptrs_48, %0 : tensor<64x64x!tt.ptr<f16>, #blocked1> loc(#loc25)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":15:13)
#loc3 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":16:13)
#loc4 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":18:14)
#loc5 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":18:32)
#loc6 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":19:14)
#loc7 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:22)
#loc8 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:14)
#loc9 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:52)
#loc10 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:22)
#loc11 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:14)
#loc12 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:52)
#loc13 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":30:19)
#loc14 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":30:9)
#loc15 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":31:19)
#loc16 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":31:9)
#loc17 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":29:16)
#loc18 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":26:5)
#loc19 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":27:13)
#loc20 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":28:13)
#loc21 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":33:9)
#loc22 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:22)
#loc23 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:14)
#loc24 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:53)
#loc25 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":37:5)
#loc38 = loc("pid_m"(#loc2))
#loc39 = loc("pid_n"(#loc3))
#loc40 = loc("offs_m"(#loc4))
#loc41 = loc("offs_m"(#loc5))
#loc42 = loc("offs_n"(#loc6))
#loc43 = loc("a_ptrs"(#loc7))
#loc44 = loc("a_ptrs"(#loc8))
#loc45 = loc("a_ptrs"(#loc9))
#loc46 = loc("b_ptrs"(#loc10))
#loc47 = loc("b_ptrs"(#loc11))
#loc48 = loc("b_ptrs"(#loc12))
#loc49 = loc("a_ptrs"(#loc13))
#loc50 = loc("a_ptrs"(#loc14))
#loc51 = loc("b_ptrs"(#loc15))
#loc52 = loc("b_ptrs"(#loc16))
#loc53 = loc("acc"(#loc17))
#loc54 = loc("a_ptrs"(#loc18))
#loc55 = loc("a"(#loc19))
#loc56 = loc("b"(#loc20))
#loc57 = loc("c"(#loc21))
#loc58 = loc("c_ptrs"(#loc22))
#loc59 = loc("c_ptrs"(#loc23))
#loc60 = loc("c_ptrs"(#loc24))
#loc61 = loc("b_ptrs"(#loc54))
#loc62 = loc("acc"(#loc61))


// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %false = arith.constant false
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
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %42 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>> -> !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %43 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c0_i32_1 = arith.constant 0 : i32
    %44 = ttg.memdesc_index %43[%c0_i32_1] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %44, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c1_i32 = arith.constant 1 : i32
    %45 = ttg.memdesc_index %43[%c1_i32] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %45, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %46:6 = scf.for %arg12 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg13 = %25, %arg14 = %37, %arg15 = %false, %arg16 = %42, %arg17 = %c0_i32_0, %arg18 = %c0_i32_0) -> (tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32)  : i32 {
      %60 = tt.load %arg13 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %61 = ttg.local_alloc %60 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %62 = tt.load %arg14 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %63 = ttg.local_alloc %62 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %true_6 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} true
      %c0_i32_7 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 0 : i32
      %c1_i32_8 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 1 : i32
      %c2_i32 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 2 : i32
      %64 = ttg.memdesc_index %43[%arg18] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
      %65 = ttng.tc_gen5_mma %61, %63, %result[%arg16], %arg15, %true, %64[%true_6] {is_async, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
      ttng.wait_barrier %64, %arg17 deps %61, %63 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %66 = tt.addptr %arg13, %39 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %67 = tt.addptr %arg14, %41 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %68 = arith.xori %arg17, %c1_i32_8 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %69 = arith.addi %arg18, %c1_i32_8 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %70 = arith.cmpi sge, %69, %c2_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %71 = arith.select %70, %c0_i32_7, %69 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %72 = arith.select %70, %68, %arg17 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      scf.yield %66, %67, %true, %65, %72, %71 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32
    } {tt.scheduled_max_stage = 1 : i32}
    %c0_i32_2 = arith.constant 0 : i32
    %47 = ttg.memdesc_index %43[%c0_i32_2] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %47 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %c1_i32_3 = arith.constant 1 : i32
    %48 = ttg.memdesc_index %43[%c1_i32_3] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %48 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %43 : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %result_4, %token_5 = ttng.tmem_load %result[%46#3] : !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>>
    %49 = arith.truncf %result_4 : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>> to tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>>
    %50 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %51 = arith.muli %50, %14 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %52 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %53 = tt.addptr %52, %51 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %54 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %55 = arith.muli %54, %32 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %56 = tt.broadcast %53 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %57 = tt.broadcast %55 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %58 = tt.addptr %56, %57 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %59 = ttg.convert_layout %49 : tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.store %58, %59 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %0 = ub.poison : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %1 = ub.poison : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %2 = ub.poison : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %3 = ub.poison : i32
    %4 = ub.poison : !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    %5 = ub.poison : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    %c2_i32 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %false = arith.constant false
    %6 = tt.get_program_id x : i32
    %7 = tt.get_program_id y : i32
    %8 = arith.muli %6, %c64_i32 : i32
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %12 = tt.splat %8 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %13 = tt.splat %8 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %14 = arith.addi %12, %9 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %15 = arith.addi %13, %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %16 = arith.muli %7, %c64_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %18 = arith.addi %17, %11 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %20 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %21 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %22 = arith.muli %19, %21 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %23 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %24 = tt.addptr %23, %22 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %27 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %28 = arith.muli %26, %27 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %29 = tt.broadcast %24 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %30 = tt.broadcast %28 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %31 = tt.addptr %29, %30 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %34 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %35 = arith.muli %33, %34 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %36 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %37 = tt.addptr %36, %35 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %38 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %39 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %40 = arith.muli %38, %39 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %41 = tt.broadcast %37 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %42 = tt.broadcast %40 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %43 = tt.addptr %41, %42 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %44 = arith.muli %arg7, %c32_i32 : i32
    %45 = tt.splat %44 : i32 -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %46 = arith.muli %arg8, %c32_i32 : i32
    %47 = tt.splat %46 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %48 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>> -> !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>
    %49 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %50 = ttg.memdesc_index %49[%c0_i32] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %50, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %51 = ttg.memdesc_index %49[%c1_i32] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %52 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %53 = tt.splat %52 : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %54 = tt.load %31, %53 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %55 = ttg.local_alloc %54 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    %56 = tt.splat %52 : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %57 = tt.load %43, %56 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %58 = ttg.local_alloc %57 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    %59 = ttg.memdesc_index %49[%c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %60 = arith.andi %52, %true : i1
    %61 = ttng.tc_gen5_mma %55, %58, %result[%48], %false, %60, %59[%true] {is_async, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %62 = arith.select %52, %61, %48 : !ttg.async.token
    %63 = tt.addptr %31, %45 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %64 = tt.addptr %43, %47 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %65 = arith.cmpi sge, %c1_i32, %c2_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %66 = arith.select %65, %c0_i32, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %67 = arith.select %65, %c1_i32, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %68 = arith.subi %arg5, %c32_i32 : i32
    %69:10 = scf.for %arg12 = %c0_i32 to %68 step %c32_i32 iter_args(%arg13 = %63, %arg14 = %64, %arg15 = %false, %arg16 = %62, %arg17 = %67, %arg18 = %66, %arg19 = %59, %arg20 = %c0_i32, %arg21 = %55, %arg22 = %58) -> (tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>)  : i32 {
      ttng.wait_barrier %arg19, %arg20 deps %arg21, %arg22 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %85 = tt.splat %true : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %86 = tt.load %arg13, %85 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %87 = ttg.local_alloc %86 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %88 = tt.splat %true : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %89 = tt.load %arg14, %88 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %90 = ttg.local_alloc %89 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %91 = ttg.memdesc_index %49[%arg18] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
      %92 = arith.andi %true, %true : i1
      %93 = ttng.tc_gen5_mma %87, %90, %result[%arg16], %true, %92, %91[%true] {is_async, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
      %94 = tt.addptr %arg13, %45 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %95 = tt.addptr %arg14, %47 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %96 = arith.xori %arg17, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %97 = arith.addi %arg18, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %98 = arith.cmpi sge, %97, %c2_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %99 = arith.select %98, %c0_i32, %97 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %100 = arith.select %98, %96, %arg17 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      scf.yield %94, %95, %true, %93, %100, %99, %91, %arg17, %87, %90 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    } {tt.scheduled_max_stage = 1 : i32}
    %70 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %71:10 = scf.if %70 -> (tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>) {
      ttng.wait_barrier %69#6, %69#7 deps %69#8, %69#9 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      scf.yield %1, %0, %true, %69#3, %3, %3, %2, %69#4, %5, %4 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    } else {
      scf.yield %69#0, %69#1, %69#2, %69#3, %69#4, %69#5, %69#6, %69#7, %69#8, %69#9 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
    }
    %72 = ttg.memdesc_index %49[%c0_i32] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %72 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %73 = ttg.memdesc_index %49[%c1_i32] : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %73 : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %49 : !ttg.memdesc<2x1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%71#3] : !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>>
    %74 = arith.truncf %result_0 : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>> to tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>>
    %75 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %76 = arith.muli %75, %20 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %77 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %78 = tt.addptr %77, %76 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %79 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %80 = arith.muli %79, %38 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %81 = tt.broadcast %78 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %82 = tt.broadcast %80 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %83 = tt.addptr %81, %82 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    %84 = ttg.convert_layout %74 : tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.store %83, %84 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
    tt.return
  }
}


