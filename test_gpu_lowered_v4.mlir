module attributes {gpu.container_module} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @main1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: !llvm.ptr, %arg19: !llvm.ptr, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(65536 : index) : i64
    %3 = llvm.mlir.constant(256 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(8 : index) : i64
    %8 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %9 = llvm.getelementptr %1[16777216] : (!llvm.ptr) -> !llvm.ptr, f32
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.add %10, %0 : i64
    %12 = llvm.call @malloc(%11) : (i64) -> !llvm.ptr
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.sub %0, %6 : i64
    %15 = llvm.add %13, %14 : i64
    %16 = llvm.urem %15, %0 : i64
    %17 = llvm.sub %15, %16 : i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr
    %19 = llvm.insertvalue %12, %8[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %18, %19[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %4, %20[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %3, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %3, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %3, %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %2, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %3, %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %6, %26[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    gpu.launch_func  @main1_kernel::@main1_kernel blocks in (%7, %6, %6) threads in (%7, %6, %6) : i64 args(%5 : i64, %4 : i64, %arg18 : !llvm.ptr, %arg19 : !llvm.ptr, %arg20 : i64, %arg21 : i64, %arg22 : i64, %arg23 : i64, %arg24 : i64, %arg25 : i64, %arg26 : i64, %4 : i64, %12 : !llvm.ptr, %18 : !llvm.ptr, %4 : i64, %3 : i64, %3 : i64, %3 : i64, %2 : i64, %3 : i64, %6 : i64, %arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : i64, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64, %arg7 : i64, %arg8 : i64, %arg9 : !llvm.ptr, %arg10 : !llvm.ptr, %arg11 : i64, %arg12 : i64, %arg13 : i64, %arg14 : i64, %arg15 : i64, %arg16 : i64, %arg17 : i64)
    llvm.return %27 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
  gpu.module @main1_kernel [#nvvm.target<chip = "sm_80">] {
    llvm.func @main1_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr, %arg22: !llvm.ptr, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: !llvm.ptr, %arg31: !llvm.ptr, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(65536 : index) : i64
      %1 = llvm.mlir.constant(0 : index) : i64
      %2 = llvm.mlir.constant(256 : index) : i64
      %3 = llvm.mlir.constant(32 : index) : i64
      %4 = llvm.mlir.constant(1 : index) : i64
      %5 = llvm.mlir.constant(4 : index) : i64
      %6 = llvm.mlir.constant(2 : index) : i64
      %7 = llvm.mlir.constant(3 : index) : i64
      %8 = nvvm.read.ptx.sreg.ctaid.x : i32
      %9 = llvm.sext %8 : i32 to i64
      %10 = nvvm.read.ptx.sreg.tid.x : i32
      %11 = llvm.sext %10 : i32 to i64
      %12 = llvm.mul %9, %arg0 overflow<nsw> : i64
      %13 = llvm.add %12, %arg1 : i64
      %14 = llvm.mul %11, %3 overflow<nsw> : i64
      llvm.br ^bb1(%1 : i64)
    ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb14
      %16 = llvm.icmp "slt" %15, %2 : i64
      llvm.cond_br %16, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      %17 = llvm.add %13, %3 : i64
      llvm.br ^bb3(%13 : i64)
    ^bb3(%18: i64):  // 2 preds: ^bb2, ^bb13
      %19 = llvm.icmp "slt" %18, %17 : i64
      llvm.cond_br %19, ^bb4, ^bb14
    ^bb4:  // pred: ^bb3
      %20 = llvm.add %14, %3 : i64
      llvm.br ^bb5(%14 : i64)
    ^bb5(%21: i64):  // 2 preds: ^bb4, ^bb12
      %22 = llvm.icmp "slt" %21, %20 : i64
      llvm.cond_br %22, ^bb6, ^bb13
    ^bb6:  // pred: ^bb5
      %23 = llvm.add %15, %3 : i64
      llvm.br ^bb7(%15 : i64)
    ^bb7(%24: i64):  // 2 preds: ^bb6, ^bb11
      %25 = llvm.icmp "slt" %24, %23 : i64
      llvm.cond_br %25, ^bb8, ^bb12
    ^bb8:  // pred: ^bb7
      %26 = llvm.mul %arg11, %0 overflow<nsw, nuw> : i64
      %27 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %28 = llvm.add %26, %27 overflow<nsw, nuw> : i64
      %29 = llvm.add %28, %24 overflow<nsw, nuw> : i64
      %30 = llvm.getelementptr inbounds|nuw %arg3[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %31 = llvm.load %30 : !llvm.ptr -> f32
      %32 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %33 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %34 = llvm.add %32, %33 overflow<nsw, nuw> : i64
      %35 = llvm.add %34, %24 overflow<nsw, nuw> : i64
      %36 = llvm.getelementptr inbounds|nuw %arg13[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %31, %36 : f32, !llvm.ptr
      llvm.br ^bb9(%1 : i64)
    ^bb9(%37: i64):  // 2 preds: ^bb8, ^bb10
      %38 = llvm.icmp "slt" %37, %2 : i64
      llvm.cond_br %38, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %39 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %40 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %41 = llvm.add %39, %40 overflow<nsw, nuw> : i64
      %42 = llvm.add %41, %37 overflow<nsw, nuw> : i64
      %43 = llvm.getelementptr inbounds|nuw %arg22[%42] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %44 = llvm.load %43 : !llvm.ptr -> f32
      %45 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %46 = llvm.mul %37, %2 overflow<nsw, nuw> : i64
      %47 = llvm.add %45, %46 overflow<nsw, nuw> : i64
      %48 = llvm.add %47, %24 overflow<nsw, nuw> : i64
      %49 = llvm.getelementptr inbounds|nuw %arg31[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %50 = llvm.load %49 : !llvm.ptr -> f32
      %51 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %52 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %53 = llvm.add %51, %52 overflow<nsw, nuw> : i64
      %54 = llvm.add %53, %24 overflow<nsw, nuw> : i64
      %55 = llvm.getelementptr inbounds|nuw %arg13[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %56 = llvm.load %55 : !llvm.ptr -> f32
      %57 = llvm.fmul %44, %50 : f32
      %58 = llvm.fadd %57, %56 : f32
      %59 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %60 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %61 = llvm.add %59, %60 overflow<nsw, nuw> : i64
      %62 = llvm.add %61, %24 overflow<nsw, nuw> : i64
      %63 = llvm.getelementptr inbounds|nuw %arg13[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %58, %63 : f32, !llvm.ptr
      %64 = llvm.add %37, %4 : i64
      %65 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %66 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %67 = llvm.add %65, %66 overflow<nsw, nuw> : i64
      %68 = llvm.add %67, %64 overflow<nsw, nuw> : i64
      %69 = llvm.getelementptr inbounds|nuw %arg22[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %70 = llvm.load %69 : !llvm.ptr -> f32
      %71 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %72 = llvm.mul %64, %2 overflow<nsw, nuw> : i64
      %73 = llvm.add %71, %72 overflow<nsw, nuw> : i64
      %74 = llvm.add %73, %24 overflow<nsw, nuw> : i64
      %75 = llvm.getelementptr inbounds|nuw %arg31[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %76 = llvm.load %75 : !llvm.ptr -> f32
      %77 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %78 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %79 = llvm.add %77, %78 overflow<nsw, nuw> : i64
      %80 = llvm.add %79, %24 overflow<nsw, nuw> : i64
      %81 = llvm.getelementptr inbounds|nuw %arg13[%80] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %82 = llvm.load %81 : !llvm.ptr -> f32
      %83 = llvm.fmul %70, %76 : f32
      %84 = llvm.fadd %83, %82 : f32
      %85 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %86 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %87 = llvm.add %85, %86 overflow<nsw, nuw> : i64
      %88 = llvm.add %87, %24 overflow<nsw, nuw> : i64
      %89 = llvm.getelementptr inbounds|nuw %arg13[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %84, %89 : f32, !llvm.ptr
      %90 = llvm.add %37, %6 : i64
      %91 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %92 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %93 = llvm.add %91, %92 overflow<nsw, nuw> : i64
      %94 = llvm.add %93, %90 overflow<nsw, nuw> : i64
      %95 = llvm.getelementptr inbounds|nuw %arg22[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %96 = llvm.load %95 : !llvm.ptr -> f32
      %97 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %98 = llvm.mul %90, %2 overflow<nsw, nuw> : i64
      %99 = llvm.add %97, %98 overflow<nsw, nuw> : i64
      %100 = llvm.add %99, %24 overflow<nsw, nuw> : i64
      %101 = llvm.getelementptr inbounds|nuw %arg31[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %102 = llvm.load %101 : !llvm.ptr -> f32
      %103 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %104 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %105 = llvm.add %103, %104 overflow<nsw, nuw> : i64
      %106 = llvm.add %105, %24 overflow<nsw, nuw> : i64
      %107 = llvm.getelementptr inbounds|nuw %arg13[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %108 = llvm.load %107 : !llvm.ptr -> f32
      %109 = llvm.fmul %96, %102 : f32
      %110 = llvm.fadd %109, %108 : f32
      %111 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %112 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %113 = llvm.add %111, %112 overflow<nsw, nuw> : i64
      %114 = llvm.add %113, %24 overflow<nsw, nuw> : i64
      %115 = llvm.getelementptr inbounds|nuw %arg13[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %110, %115 : f32, !llvm.ptr
      %116 = llvm.add %37, %7 : i64
      %117 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %118 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %119 = llvm.add %117, %118 overflow<nsw, nuw> : i64
      %120 = llvm.add %119, %116 overflow<nsw, nuw> : i64
      %121 = llvm.getelementptr inbounds|nuw %arg22[%120] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %122 = llvm.load %121 : !llvm.ptr -> f32
      %123 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %124 = llvm.mul %116, %2 overflow<nsw, nuw> : i64
      %125 = llvm.add %123, %124 overflow<nsw, nuw> : i64
      %126 = llvm.add %125, %24 overflow<nsw, nuw> : i64
      %127 = llvm.getelementptr inbounds|nuw %arg31[%126] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %128 = llvm.load %127 : !llvm.ptr -> f32
      %129 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %130 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %131 = llvm.add %129, %130 overflow<nsw, nuw> : i64
      %132 = llvm.add %131, %24 overflow<nsw, nuw> : i64
      %133 = llvm.getelementptr inbounds|nuw %arg13[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %134 = llvm.load %133 : !llvm.ptr -> f32
      %135 = llvm.fmul %122, %128 : f32
      %136 = llvm.fadd %135, %134 : f32
      %137 = llvm.mul %18, %0 overflow<nsw, nuw> : i64
      %138 = llvm.mul %21, %2 overflow<nsw, nuw> : i64
      %139 = llvm.add %137, %138 overflow<nsw, nuw> : i64
      %140 = llvm.add %139, %24 overflow<nsw, nuw> : i64
      %141 = llvm.getelementptr inbounds|nuw %arg13[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %136, %141 : f32, !llvm.ptr
      %142 = llvm.add %37, %5 : i64
      llvm.br ^bb9(%142 : i64)
    ^bb11:  // pred: ^bb9
      %143 = llvm.add %24, %4 : i64
      llvm.br ^bb7(%143 : i64)
    ^bb12:  // pred: ^bb7
      %144 = llvm.add %21, %4 : i64
      llvm.br ^bb5(%144 : i64)
    ^bb13:  // pred: ^bb5
      %145 = llvm.add %18, %4 : i64
      llvm.br ^bb3(%145 : i64)
    ^bb14:  // pred: ^bb3
      %146 = llvm.add %15, %3 : i64
      llvm.br ^bb1(%146 : i64)
    ^bb15:  // pred: ^bb1
      llvm.return
    }
  }
}

