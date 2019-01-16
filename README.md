
<div align="center">
 <img src="modelnet_example.png"><br><br>
</div>

-----------------


## ILA-SCNN

This repository contains the source code of our paper: [Inference, Learning and Attention Mechanisms that Exploit and Preserve Sparsity in Convolutional Networks](https://arxiv.org/abs/1801.10585)
Our implementation extents tensorflow by providing GPU implementations for layers that are able efficiently process large-scale, sparse data. For more details please refer to our paper.


## Installation

After cloning this repository:
1) Follow: [Installing TensorFlow](https://www.tensorflow.org/install/install_sources) to install tensorflow from source code.

2) build our sparse modules:
```
bazel build -c opt --config=cuda --strip=never -s //tensorflow/core/user_ops:direct_sparse_conv_kd.so
```
3) set environment variables
```
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$HOME/src/ILA-SCNN/bazel-bin/tensorflow/core/user_ops"' >> ~/.bashrc
echo 'export PYTHONPATH="$PYTHONPATH:$HOME/src/ILA-SCNN/tensorflow/core/user_ops"' >> ~/.bashrc
source ~/.bashrc
```
Learn more examples about how to do specific tasks in TensorFlow at the [tutorials page of tensorflow.org](https://www.tensorflow.org/tutorials/).

## Examples
We provide the following examples:
1)  [Modelnet Application](https://github.com/TimoHackel/ILA-SCNN/tree/master/tensorflow/core/user_ops/direct_sparse_experiments/modelnet)
2) [MNIST Application](https://github.com/TimoHackel/ILA-SCNN/tree/master/tensorflow/core/user_ops/direct_sparse_experiments/mnist)

The correct data sets will be downloaded automatically when running the scripts.
