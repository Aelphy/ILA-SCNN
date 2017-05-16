#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


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


REGISTER_OP("DirectSparseToDenseGrad")
  .Attr("T: realnumbertype")
  .Attr("Tindices: realnumbertype")
  .Input("in_indices: Tindices")
  .Input("gradients: T")
  .Output("backprops: T");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T, typename Tindices>
class DirectSparseToDenseGrad : public OpKernel {
 public:
  explicit DirectSparseToDenseGrad(OpKernelConstruction* context) : OpKernel(context){}



  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *gradients;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    auto in_ind = in_indices->matrix<Tindices>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto grads = gradients->flat<T>();

    //allocate output data
    Tensor *backprops = NULL;
    TensorShape backprops_shape = {in_ind.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", backprops_shape, &backprops));
    auto bp = backprops->flat<T>();

    auto in_ind_ptr = &in_ind; auto grads_ptr = &grads; auto bp_ptr = &bp;
#pragma omp parallel for firstprivate(bp_ptr)
    for(int64 i = 0; i < bp_ptr->dimension(0); ++i){
      (*bp_ptr)(i) = 0;
    }
    
#pragma omp parallel for firstprivate(in_ind_ptr, grads_ptr, bp_ptr)
    for(int64 i = 0; i < in_ind_ptr->dimension(0); ++i){
      (*bp_ptr)(i) = (*grads_ptr)((*in_ind_ptr)(i));
    }
  }
};

#define REGISTER_CPU(type, index_type)                                   \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseToDenseGrad").Device(DEVICE_CPU).TypeConstraint<type>("T").TypeConstraint<index_type>("Tindices"), \
                          DirectSparseToDenseGrad<CPUDevice, type, index_type>);

REGISTER_CPU(float, int64);
REGISTER_CPU(double, int64);
REGISTER_CPU(int32, int64);
REGISTER_CPU(complex64, int64);
REGISTER_CPU(complex128, int64);
#undef REGISTER_CPU

