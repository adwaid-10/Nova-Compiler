module {
  func.func @constant_fold() -> tensor<1xf32> {
  %c1 = nova.constant {value = dense<[1.0]> : tensor<1xf32>} : tensor<1xf32>
  %c2 = nova.constant {value = dense<[2.0]> : tensor<1xf32>} : tensor<1xf32>
  %res = nova.add %c1, %c2 : tensor<1xf32>, tensor<1xf32>
  return %res : tensor<1xf32>
}
}