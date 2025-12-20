func.func @f(%input : memref<10xf32>, %output : memref<10xf32>, %reduc : memref<10xf32>) {
  %zero = arith.constant 0. : f32
  %one = arith.constant 1. : f32
  affine.for %i = 0 to 10 {
    %0 = affine.load %input[%i] : memref<10xf32>
    %2 = arith.addf %0, %one : f32
    affine.store %2, %output[%i] : memref<10xf32>
  }
  affine.for %i = 0 to 10 {
    %0 = affine.load %input[%i] : memref<10xf32>
    %1 = arith.addf %0, %zero : f32
    affine.store %1, %reduc[%i] : memref<10xf32>
  }
  return 
}

//func.func @f_nova(%input : tensor<10xf32>) -> (tensor<10xf32>, tensor<1xf32>) {
//  %one = nova.constant {value = dense<[1.0]> : tensor<1xf32>} : tensor<1xf32>
//  %zero = nova.constant {value = dense<[0.0]> : tensor<1xf32>} : tensor<1xf32>
//  %output = nova.add %input, %one : tensor<10xf32>, tensor<1xf32>
//  %reduc = nova.add %input, %zero : tensor<10xf32>, tensor<1xf32>
//  return %output, %reduc : tensor<10xf32>, tensor<10xf32>
//}

