func.func @f_nova(%input : tensor<10xf32>,%input2:tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %one = nova.constant {value = dense<[1.0]> : tensor<1xf32>} : tensor<1xf32>
  %zero = nova.constant {value = dense<[0.0]> : tensor<1xf32>} : tensor<1xf32>
  %output = nova.add %input, %one : tensor<10xf32>, tensor<1xf32>
  %reduc = nova.add %input, %input2 : tensor<10xf32>, tensor<10xf32>
  return %output, %reduc : tensor<10xf32>, tensor<10xf32>
}