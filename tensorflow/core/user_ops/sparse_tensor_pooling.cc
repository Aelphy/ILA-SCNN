#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_pooling.h"


/** SparseTensorMaxPooling
  * \ingroup CXX11_NeuralNetworks_Module
  * 
  * \brief Applies a 3D convolution over a multichannel input voxel block.
  * 
  * The input parameter is expected to be a tensor with a rank of 4 or more (channels, depth, height, width, and optionally others).
  * The kernel parameter is expected to be a 5D tensor (filters, channels, kernel_depth, kernel_height, kernel_width).
  * 
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be filters, depth, height, width (and others if applicable).
  */


REGISTER_OP("SparseTensorMaxPooling")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("pooling_shape: int64")
  .Output("out_indices: int64")
  .Output("out_values: T")
  .Output("out_shape: int64")
  .Output("out_corresponding_indices: int64")
  .Attr("dim: int = 3");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T>
class SparseTensorPooling : public OpKernel {
 public:
  explicit SparseTensorPooling(OpKernelConstruction* context) : OpKernel(context) 
  {
    OP_REQUIRES_OK(context, context->GetAttr("dim", &dim));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *pooling_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("pooling_shape", &pooling_shape));
    auto in_ind = in_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    auto p_sh = pooling_shape->flat<int64>();
    std::vector<std::vector<int64> > output_ids;
    std::vector<int64> corresponding_output_ids;
    std::vector<T> output_vals;
    std::vector<int64> output_shape;
    max_pooling<Device, T, decltype(in_ind), decltype(in_vals), decltype(in_sh)> (in_ind, 
                                                                                  in_vals, 
                                                                                  in_sh, 
                                                                                  p_sh, 
                                                                                  output_ids, 
                                                                                  output_vals, 
                                                                                  output_shape, 
                                                                                  corresponding_output_ids, 
                                                                                  dim);
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL, *sparse_cind;
    TensorShape out_ind_shape = {(int64) output_ids.size(), (int64) in_ind.dimension(1)};
    TensorShape out_val_shape = {(int64) output_ids.size()};
    TensorShape out_sh_shape = {(int64) in_ind.dimension(1)};
    TensorShape out_cind_shape = {(int64) output_ids.size()};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &sparse_shape));
    OP_REQUIRES_OK(context, context->allocate_output("out_corresponding_indices", out_cind_shape, &sparse_cind));

    auto out_ind = sparse_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto out_vals = sparse_values->flat<T>();
    auto out_sh = sparse_shape->flat<int64>();
    auto out_cind = sparse_cind->flat<int64>();

    auto out_ind_ptr = &out_ind; auto out_vals_ptr = &out_vals; auto out_cind_ptr = &out_cind; 
    auto output_keys_ptr = &output_ids; auto output_values_ptr = &output_vals; auto output_cids_ptr = &corresponding_output_ids;
#pragma omp parallel for firstprivate(output_keys_ptr, output_values_ptr, output_cids_ptr, out_vals_ptr, out_ind_ptr, out_cind_ptr)
    for(auto i = 0; i < (*output_keys_ptr).size(); ++i){
        for(int64 j = 0; j < (*output_keys_ptr)[i].size(); ++j){
          (*out_ind_ptr)(i,j) = (*output_keys_ptr)[i][j];
        }
        (*out_vals_ptr)(i) = (*output_values_ptr)[i];
        (*out_cind_ptr)(i) = (*output_cids_ptr)[i];
    }
    for(int64 idx = 0; idx < in_ind.dimension(1); ++idx){
        out_sh(idx) = output_shape[idx];
    }
  }
  private:
    int dim;
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorMaxPooling").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
                          SparseTensorPooling<CPUDevice, type>);

REGISTER_CPU(float);
//REGISTER_CPU(double);
//REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);
#undef REGISTER_CPU

