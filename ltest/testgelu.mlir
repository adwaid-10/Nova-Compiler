module {
  func.func @main(%arg0: tensor<1x8x3xi32>) -> tensor<1x8x3xf32> {
    %1=nova.gelu %arg0 : tensor<1x8x3xi32> 
 return %1 :tensor<1x8x3xf32>
 }
}
module {
  func.func @main(%arg0: tensor<1x8x3xi32>) -> tensor<1x8x3xf32> {
    %0 = tosa.cast %arg0 : (tensor<1x8x3xi32>) -> tensor<1x8x3xf32>
    %1 = nova.constant {value = dense<3.000000e+00> : tensor<1x8x3xf32>} : tensor<1x8x3xf32>
    %2 = tosa.pow %0, %1 : (tensor<1x8x3xf32>, tensor<1x8x3xf32>) -> tensor<1x8x3xf32>
    %3 = nova.constant {value = dense<4.471500e-02> : tensor<1x8x3xf32>} : tensor<1x8x3xf32>
    %4 = nova.mul %2, %3 : tensor<1x8x3xf32>, tensor<1x8x3xf32>
    %5 = tosa.add %0, %4 : (tensor<1x8x3xf32>, tensor<1x8x3xf32>) -> tensor<1x8x3xf32>
    %6 = nova.constant {value = dense<0.797884583> : tensor<1x8x3xf32>} : tensor<1x8x3xf32>
    %7 = nova.mul %5, %6 : tensor<1x8x3xf32>, tensor<1x8x3xf32>
    %8 = tosa.tanh %7 : (tensor<1x8x3xf32>) -> tensor<1x8x3xf32>
    %9 = nova.constant {value = dense<1.000000e+00> : tensor<1x8x3xf32>} : tensor<1x8x3xf32>
    %10 = tosa.add %8, %9 : (tensor<1x8x3xf32>, tensor<1x8x3xf32>) -> tensor<1x8x3xf32>
    %11 = nova.constant {value = dense<5.000000e-01> : tensor<1x8x3xf32>} : tensor<1x8x3xf32>
    %12 = nova.mul %0, %11 : tensor<1x8x3xf32>, tensor<1x8x3xf32>
    %13 = nova.mul %12, %10 : tensor<1x8x3xf32>, tensor<1x8x3xf32>
    return %13 : tensor<1x8x3xf32>
  }
}
