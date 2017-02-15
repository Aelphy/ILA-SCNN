#include <map>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("SparseTensorDenseConv3D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("strides: list(int) >= 5")
    .Doc(R"doc(
Computes a 3-D convolution given sparse 5-D `input` and  dense`filter` tensors.

In signal processing, cross-correlation is a measure of similarity of
two waveforms as a function of a time-lag applied to one of them. This
is also known as a sliding dot product or sliding inner-product.

Our Conv3D implements a form of cross-correlation.

input: Shape `[batch, in_depth, in_height, in_width, in_channels]`.
filter: Shape `[filter_depth, filter_height, filter_width, in_channels,
  out_channels]`. `in_channels` must match between `input` and `filter`.
strides: 1-D tensor of length 5. The stride of the sliding window for each
  dimension of `input`. Must have `strides[0] = strides[4] = 1`.
padding is automatically set to zero padding to keep the tensor sparse
)doc");


#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class SparseTensorDenseConv3D : public OpKernel {
 public:
  explicit SparseTensorDenseConv3D(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& filter = context->input(1);
    auto input = input_tensor.flat<int32>();
    std::vector<int32> stride;
    //OP_REQUIRES_OK(context, context->GetAttr("strides", &stride));

    /*
    for(size_t i = 0; i < input_indices.size(); ++i){
      if(index_is_unaffected_by_stride(input_indices[i], stride)){
          update_indices, update_values = dense_filter_update_output(input_indices[i], input_values[i], filter);
          SparseAddOp(update_indices, update_values)
      }
    }*/
    


  }

 private:  
  /*inline bool index_is_unaffected_by_stride(const std::vector<int32>& id, const std::vector<int32>& stride) const {
    assert(id.size() == stride.size());
    for(int32 i = 0; i < stride.size(); ++i){
      if(stride[i] > 0){
        if(((id[i] + 1) % stride[i]) == 0){
          return false;
        }
      }
    }
    return true;
  }

  inline void TTypes<Index>::Matrix indices dense_filter_update_output(TTypes<Index>::Matrix& a_conv_filter, TTypes<Index>::Matrix& a_input_indices, ){

  }
  */
};


REGISTER_KERNEL_BUILDER(Name("SparseTensorDenseConv3D").Device(DEVICE_CPU), SparseTensorDenseConv3D);
