#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_pooling.h"


/** SparseTensorMaxPoolingGrad
  * \ingroup CXX11_NeuralNetworks_Module
  * 
  * \brief Applies a 3D convolution over a multichannel input voxel block.
  * 
  * The input parameter is expected to be a tensor with a rank of 4 or more (channels, depth, height, width, and optionally others).
  * The kernel parameter is expected to be a 5D tensor (filters, channels, kernel_depth, kernel_height, kernel_width).
  * 
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be filters, depth, height, width (and others if applicable).
  */


REGISTER_OP("SparseTensorMaxPoolingGrad")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("gradients: T")
  .Input("corresponding_indices: int64")
  .Output("backprops: T");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T>
class SparseTensorPoolingGrad : public OpKernel {
 public:
  explicit SparseTensorPoolingGrad(OpKernelConstruction* context) : OpKernel(context){}



  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *gradients, *corresponding_ind;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    OP_REQUIRES_OK(context, context->input("corresponding_indices", &corresponding_ind));
    auto in_ind = in_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto grads = gradients->flat<T>();
    auto cor_ind = corresponding_ind->flat<int64>();

    //allocate output data
    Tensor *backprops = NULL;
    TensorShape backprops_shape = {in_ind.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", backprops_shape, &backprops));
    auto bp = backprops->flat<T>();

    auto in_ind_ptr = &in_ind; auto grads_ptr = &grads; auto cor_ind_ptr = &cor_ind; auto bp_ptr = &bp;
#pragma omp parallel for firstprivate(bp)
    for(int64 i = 0; i < bp_ptr->dimension(0); ++i){
      (*bp_ptr)(i) = 0;
    }
    
#pragma omp parallel for firstprivate(in_ind_ptr, grads_ptr, cor_ind_ptr, bp_ptr)
    for(int64 i = 0; i < cor_ind_ptr->dimension(0); ++i){
      (*bp_ptr)((*cor_ind_ptr)(i)) = (*grads_ptr)(i);
    }
  }
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorMaxPoolingGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
                          SparseTensorPoolingGrad<CPUDevice, type>);

REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);
#undef REGISTER_CPU

