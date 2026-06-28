// -----// IR Dump Before TritonGPUPipeline: tritongpu-pipeline{dump-intermediate-steps=true num-stages=3} ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>
#loc = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":6:1)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>
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
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("c_ptr"(#loc)), %M: i32 loc("M"(#loc)), %N: i32 loc("N"(#loc)), %K: i32 loc("K"(#loc)), %stride_am: i32 loc("stride_am"(#loc)), %stride_ak: i32 loc("stride_ak"(#loc)), %stride_bk: i32 loc("stride_bk"(#loc)), %stride_bn: i32 loc("stride_bn"(#loc)), %stride_cm: i32 loc("stride_cm"(#loc)), %stride_cn: i32 loc("stride_cn"(#loc))) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #linear> loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %false = arith.constant false loc(#loc1)
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
    %acc, %acc_36 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token) loc(#loc55)
    %acc_37 = ttng.tmem_store %cst, %acc[%acc_36], %true : tensor<64x64xf32, #linear> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc55)
    %acc_38:4 = scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%b_ptrs_50 = %b_ptrs_31, %a_ptrs_51 = %a_ptrs_20, %acc_52 = %false, %acc_53 = %acc_37) -> (tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<64x32x!tt.ptr<f16>, #blocked>, i1, !ttg.async.token)  : i32 {
      %a = tt.load %a_ptrs_51 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked> loc(#loc57)
      %a_54 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem> loc(#loc57)
      %b = tt.load %b_ptrs_50 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked2> loc(#loc58)
      %b_55 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #blocked2>) -> !ttg.memdesc<32x64xf16, #shared1, #smem> loc(#loc58)
      %acc_56 = ttng.tc_gen5_mma %a_54, %b_55, %acc[%acc_53], %acc_52, %true {loop.cluster = 0 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<64x32xf16, #shared, #smem>, !ttg.memdesc<32x64xf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> loc(#loc55)
      %a_ptrs_57 = tt.addptr %a_ptrs_51, %a_ptrs_33 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked> loc(#loc52)
      %b_ptrs_58 = tt.addptr %b_ptrs_50, %b_ptrs_35 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<32x64xi32, #blocked2> loc(#loc54)
      scf.yield %b_ptrs_58, %a_ptrs_57, %true, %acc_56 : tensor<32x64x!tt.ptr<f16>, #blocked2>, tensor<64x32x!tt.ptr<f16>, #blocked>, i1, !ttg.async.token loc(#loc19)
    } {tt.scheduled_max_stage = 0 : i32} loc(#loc64)
    %acc_39, %acc_40 = ttng.tmem_load %acc[%acc_38#3] : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear> loc(#loc55)
    %c = arith.truncf %acc_39 : tensor<64x64xf32, #linear> to tensor<64x64xf16, #linear> loc(#loc59)
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<64x1xi32, #blocked2> loc(#loc60)
    %c_ptrs_41 = arith.muli %c_ptrs, %a_ptrs_10 : tensor<64x1xi32, #blocked2> loc(#loc60)
    %c_ptrs_42 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked2> loc(#loc61)
    %c_ptrs_43 = tt.addptr %c_ptrs_42, %c_ptrs_41 : tensor<64x1x!tt.ptr<f16>, #blocked2>, tensor<64x1xi32, #blocked2> loc(#loc61)
    %c_ptrs_44 = tt.splat %stride_cn : i32 -> tensor<1x64xi32, #blocked2> loc(#loc62)
    %c_ptrs_45 = ttg.convert_layout %b_ptrs_25 : tensor<1x64xi32, #blocked1> -> tensor<1x64xi32, #blocked2> loc(#loc62)
    %c_ptrs_46 = arith.muli %c_ptrs_44, %c_ptrs_45 : tensor<1x64xi32, #blocked2> loc(#loc62)
    %c_ptrs_47 = tt.broadcast %c_ptrs_43 : tensor<64x1x!tt.ptr<f16>, #blocked2> -> tensor<64x64x!tt.ptr<f16>, #blocked2> loc(#loc61)
    %c_ptrs_48 = tt.broadcast %c_ptrs_46 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2> loc(#loc61)
    %c_ptrs_49 = tt.addptr %c_ptrs_47, %c_ptrs_48 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2> loc(#loc61)
    %0 = ttg.convert_layout %c : tensor<64x64xf16, #linear> -> tensor<64x64xf16, #blocked2> loc(#loc26)
    tt.store %c_ptrs_49, %0 : tensor<64x64x!tt.ptr<f16>, #blocked2> loc(#loc26)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":15:13)
#loc3 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":16:13)
#loc4 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":18:14)
#loc5 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":18:32)
#loc6 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":19:14)
#loc7 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":20:14)
#loc8 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:22)
#loc9 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:14)
#loc10 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":22:52)
#loc11 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:22)
#loc12 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:14)
#loc13 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":23:52)
#loc14 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":30:19)
#loc15 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":30:9)
#loc16 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":31:19)
#loc17 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":31:9)
#loc18 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":29:16)
#loc19 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":26:5)
#loc20 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":27:13)
#loc21 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":28:13)
#loc22 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":33:9)
#loc23 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:22)
#loc24 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:14)
#loc25 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":36:53)
#loc26 = loc("/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul.py":37:5)
#loc39 = loc("pid_m"(#loc2))
#loc40 = loc("pid_n"(#loc3))
#loc41 = loc("offs_m"(#loc4))
#loc42 = loc("offs_m"(#loc5))
#loc43 = loc("offs_n"(#loc6))
#loc44 = loc("offs_k"(#loc7))
#loc45 = loc("a_ptrs"(#loc8))
#loc46 = loc("a_ptrs"(#loc9))
#loc47 = loc("a_ptrs"(#loc10))
#loc48 = loc("b_ptrs"(#loc11))
#loc49 = loc("b_ptrs"(#loc12))
#loc50 = loc("b_ptrs"(#loc13))
#loc51 = loc("a_ptrs"(#loc14))
#loc52 = loc("a_ptrs"(#loc15))
#loc53 = loc("b_ptrs"(#loc16))
#loc54 = loc("b_ptrs"(#loc17))
#loc55 = loc("acc"(#loc18))
#loc56 = loc("a_ptrs"(#loc19))
#loc57 = loc("a"(#loc20))
#loc58 = loc("b"(#loc21))
#loc59 = loc("c"(#loc22))
#loc60 = loc("c_ptrs"(#loc23))
#loc61 = loc("c_ptrs"(#loc24))
#loc62 = loc("c_ptrs"(#loc25))
#loc63 = loc("b_ptrs"(#loc56))
#loc64 = loc("acc"(#loc63))


// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>>
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %false = arith.constant false
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
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %43 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>> -> !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %44 = ttg.local_alloc : () -> !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %c0_i32_1 = arith.constant 0 : i32
    %45 = ttg.memdesc_index %44[%c0_i32_1] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %45, 1 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %c1_i32 = arith.constant 1 : i32
    %46 = ttg.memdesc_index %44[%c1_i32] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %46, 1 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %47:6 = scf.for %arg12 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg13 = %38, %arg14 = %26, %arg15 = %false, %arg16 = %43, %arg17 = %c0_i32_0, %arg18 = %c0_i32_0) -> (tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32)  : i32 {
      %62 = tt.load %arg14 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %63 = ttg.local_alloc %62 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>
      %64 = tt.load %arg13 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %65 = ttg.local_alloc %64 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      %true_6 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} true
      %c0_i32_7 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 0 : i32
      %c1_i32_8 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 1 : i32
      %c2_i32 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 2 : i32
      %66 = ttg.memdesc_index %44[%arg18] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
      %67 = ttng.tc_gen5_mma %63, %65, %result[%arg16], %arg15, %true, %66[%true_6] {is_async, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
      ttng.wait_barrier %66, %arg17 deps %63, %65 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      %68 = tt.addptr %arg14, %40 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %69 = tt.addptr %arg13, %42 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %70 = arith.xori %arg17, %c1_i32_8 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %71 = arith.addi %arg18, %c1_i32_8 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %72 = arith.cmpi sge, %71, %c2_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %73 = arith.select %72, %c0_i32_7, %71 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %74 = arith.select %72, %70, %arg17 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      scf.yield %69, %68, %true, %67, %74, %73 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32
    } {tt.scheduled_max_stage = 1 : i32}
    %c0_i32_2 = arith.constant 0 : i32
    %48 = ttg.memdesc_index %44[%c0_i32_2] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %48 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %c1_i32_3 = arith.constant 1 : i32
    %49 = ttg.memdesc_index %44[%c1_i32_3] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %44 : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %result_4, %token_5 = ttng.tmem_load %result[%47#3] : !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>>
    %50 = arith.truncf %result_4 : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>> to tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>>
    %51 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %52 = arith.muli %51, %16 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %53 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %54 = tt.addptr %53, %52 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %55 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %56 = ttg.convert_layout %32 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %57 = arith.muli %55, %56 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %58 = tt.broadcast %54 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %59 = tt.broadcast %57 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %60 = tt.addptr %58, %59 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %61 = ttg.convert_layout %50 : tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.store %60, %61 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %0 = ub.poison : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %1 = ub.poison : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %2 = ub.poison : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %3 = ub.poison : i32
    %4 = ub.poison : !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
    %5 = ub.poison : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>
    %c2_i32 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32} 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>>
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %false = arith.constant false
    %6 = tt.get_program_id x : i32
    %7 = tt.get_program_id y : i32
    %8 = arith.muli %6, %c64_i32 : i32
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %12 = tt.splat %8 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %13 = tt.splat %8 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %14 = arith.addi %12, %9 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %15 = arith.addi %13, %11 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %16 = arith.muli %7, %c64_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %18 = arith.addi %17, %10 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>>
    %19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>>
    %20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>>
    %21 = tt.expand_dims %14 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %22 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %23 = tt.splat %arg6 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %24 = arith.muli %21, %23 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %25 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %26 = tt.addptr %25, %24 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %27 = tt.expand_dims %20 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %28 = tt.splat %arg7 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %29 = arith.muli %27, %28 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %30 = tt.broadcast %26 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>> -> tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %31 = tt.broadcast %29 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>> -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %32 = tt.addptr %30, %31 : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %33 = tt.expand_dims %19 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %34 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %35 = arith.muli %33, %34 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %36 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %37 = tt.addptr %36, %35 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %38 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %39 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %40 = arith.muli %38, %39 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>>
    %41 = tt.broadcast %37 : tensor<32x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %42 = ttg.convert_layout %40 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %43 = tt.broadcast %42 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %44 = tt.addptr %41, %43 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %45 = arith.muli %arg7, %c32_i32 : i32
    %46 = tt.splat %45 : i32 -> tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %47 = arith.muli %arg8, %c32_i32 : i32
    %48 = tt.splat %47 : i32 -> tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %49 = ttng.tmem_store %cst, %result[%token], %true : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>> -> !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>
    %50 = ttg.local_alloc : () -> !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %51 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %52 = ttg.memdesc_index %50[%c1_i32] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.init_barrier %52, 1 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %53 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %54 = tt.splat %53 : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %55 = tt.load %32, %54 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %56 = ttg.local_alloc %55 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>
    %57 = tt.splat %53 : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %58 = tt.load %44, %57 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %59 = ttg.local_alloc %58 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
    %60 = ttg.memdesc_index %50[%c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %61 = arith.andi %53, %true : i1
    %62 = ttng.tc_gen5_mma %56, %59, %result[%49], %false, %61, %60[%true] {is_async, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %63 = arith.select %53, %62, %49 : !ttg.async.token
    %64 = tt.addptr %32, %46 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
    %65 = tt.addptr %44, %48 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %66 = arith.cmpi sge, %c1_i32, %c2_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %67 = arith.select %66, %c0_i32, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %68 = arith.select %66, %c1_i32, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
    %69 = arith.subi %arg5, %c32_i32 : i32
    %70:10 = scf.for %arg12 = %c0_i32 to %69 step %c32_i32 iter_args(%arg13 = %65, %arg14 = %64, %arg15 = %false, %arg16 = %63, %arg17 = %68, %arg18 = %67, %arg19 = %60, %arg20 = %c0_i32, %arg21 = %56, %arg22 = %59) -> (tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>)  : i32 {
      ttng.wait_barrier %arg19, %arg20 deps %arg21, %arg22 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      %87 = tt.splat %true : i1 -> tensor<64x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %88 = tt.load %arg14, %87 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %89 = ttg.local_alloc %88 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<64x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>) -> !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>
      %90 = tt.splat %true : i1 -> tensor<32x64xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %91 = tt.load %arg13, %90 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %92 = ttg.local_alloc %91 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : (tensor<32x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>) -> !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      %93 = ttg.memdesc_index %50[%arg18] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
      %94 = arith.andi %true, %true : i1
      %95 = ttng.tc_gen5_mma %89, %92, %result[%arg16], %true, %94, %93[%true] {is_async, loop.cluster = 1 : i32, loop.stage = 0 : i32} : !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
      %96 = tt.addptr %arg14, %46 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, tensor<64x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>
      %97 = tt.addptr %arg13, %48 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<32x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
      %98 = arith.xori %arg17, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %99 = arith.addi %arg18, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %100 = arith.cmpi sge, %99, %c2_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %101 = arith.select %100, %c0_i32, %99 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %102 = arith.select %100, %98, %arg17 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      scf.yield %97, %96, %true, %95, %102, %101, %93, %arg17, %89, %92 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
    } {tt.scheduled_max_stage = 1 : i32}
    %71 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %72:10 = scf.if %71 -> (tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>) {
      ttng.wait_barrier %70#6, %70#7 deps %70#8, %70#9 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
      scf.yield %0, %1, %true, %70#3, %3, %3, %2, %70#4, %5, %4 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
    } else {
      scf.yield %70#0, %70#1, %70#2, %70#3, %70#4, %70#5, %70#6, %70#7, %70#8, %70#9 : tensor<32x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>>, i1, !ttg.async.token, i32, i32, !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>, i32, !ttg.memdesc<64x32xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>, #ttg.shared_memory>, !ttg.memdesc<32x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CGALayout = [[0, 1]]}>, #ttg.shared_memory>
    }
    %73 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %73 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %74 = ttg.memdesc_index %50[%c1_i32] : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttng.inval_barrier %74 : !ttg.memdesc<2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %50 : !ttg.memdesc<2x2xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>, #ttg.shared_memory, mutable>
    %result_0, %token_1 = ttng.tmem_load %result[%72#3] : !ttg.memdesc<64x64xf32, #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>>
    %75 = arith.truncf %result_0 : tensor<64x64xf32, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>> to tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>>
    %76 = tt.splat %arg10 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %77 = arith.muli %76, %22 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %78 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %79 = tt.addptr %78, %77 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %80 = tt.splat %arg11 : i32 -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %81 = ttg.convert_layout %38 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[0, 1]]}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %82 = arith.muli %80, %81 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %83 = tt.broadcast %79 : tensor<64x1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %84 = tt.broadcast %82 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %85 = tt.addptr %83, %84 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    %86 = ttg.convert_layout %75 : tensor<64x64xf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]], warp = [[16, 0], [32, 0]], block = [[0, 32]]}>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.store %85, %86 : tensor<64x64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>>
    tt.return
  }
}


