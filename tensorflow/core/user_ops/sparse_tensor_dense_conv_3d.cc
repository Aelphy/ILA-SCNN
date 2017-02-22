#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_sparse_kernel_dense_conv_3d.h"


/** SparseTensorSparseKernelDenseConv3D
  * \ingroup CXX11_NeuralNetworks_Module
  * 
  * \brief Applies a 3D convolution over a multichannel input voxel block.
  * 
  * The input parameter is expected to be a tensor with a rank of 4 or more (channels, depth, height, width, and optionally others).
  * The kernel parameter is expected to be a 5D tensor (filters, channels, kernel_depth, kernel_height, kernel_width).
  * 
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be filters, depth, height, width (and others if applicable).
  */



//TODO: How do I use REGISTER_OP with parameter T?
//  .Attr("T: {float, double, int32, complex64, complex128}")
REGISTER_OP("SparseTensorDenseConv3D")
  .Attr("T: {float}")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("dense_filter: T")
  .Output("sparse_indices: int64")
  .Output("sparse_values: T")
  .Output("sparse_shape: int64")
  .Attr("strides: list(int) >= 5");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template<typename T, int N>
class Converter{
 public:
  void convert(const Tensor* t_dense, Tensor* s_ind, Tensor* s_val, const T&  epsilon) const { /*generic case not implemented*/};
};

template<typename T>
class Converter<T, 5>{
 public:
  void convert(const Tensor* t_dense, Tensor* s_ind, Tensor* s_val, const T&  epsilon) const {
    auto out_ind = s_ind->matrix<int64>(); 
    auto out_vals = s_val->flat<T>();
    auto d_data = t_dense->tensor<T, 5>();
    int64 id_cnt=0;
    for(int64 i = 0;  i < d_data.dimension(0); ++i){
      for(int64 j = 0;  j < d_data.dimension(1); ++j){
        for(int64 k = 0;  k < d_data.dimension(2); ++k){
          for(int64 l = 0;  l < d_data.dimension(3); ++l){
            for(int64 m = 0;  m < d_data.dimension(4); ++m){
              if(d_data(i,j,k,l,m) > epsilon || d_data(i,j,k,l,m) < epsilon){
                id_cnt++;
                out_ind(id_cnt, 0) = i; out_ind(id_cnt, 1) = j; out_ind(id_cnt, 2) = k; out_ind(id_cnt, 3) = l;
                out_vals(id_cnt) = d_data(i,j,k,l,m);
              }
            }
          }
        }
      }
    }
  };
};

template<typename T>
class Converter<T, 4>{
 public:
  void convert(const Tensor* t_dense, Tensor* s_ind, Tensor* s_val, const T&  epsilon) const {
    auto out_ind = s_ind->matrix<int64>(); 
    auto out_vals = s_val->flat<T>();
    auto d_data = t_dense->tensor<T, 4>();
    int64 id_cnt=0;
    for(int64 i = 0;  i < d_data.dimension(0); ++i){
     for(int64 j = 0;  j < d_data.dimension(1); ++j){
        for(int64 k = 0;  k < d_data.dimension(2); ++k){
          for(int64 l = 0;  l < d_data.dimension(3); ++l){

            if(d_data(i,j,k,l) > epsilon || d_data(i,j,k,l) < epsilon){
              id_cnt++;
              out_ind(id_cnt, 0) = i; out_ind(id_cnt, 1) = j; out_ind(id_cnt, 2) = k; out_ind(id_cnt, 3) = l;
              out_vals(id_cnt) = d_data(i,j,k,l);
            }
          }
        }
      }
    }
  };
};

template <typename Device, typename T>
class SparseTensorDenseConv3D : public OpKernel {
 public:
  explicit SparseTensorDenseConv3D(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *dense_filter;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("dense_filter", &dense_filter));
    auto in_ind = in_indices->matrix<int64>(); //[batch, depth, height, width, in_channels] //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    std::map<std::vector<int64>, T> output_map; //stores the values for the output tensor
    Tensor *filter_indices = NULL, *filter_values = NULL, *filter_shape = NULL;
    dense_to_sparse(dense_filter, filter_indices, filter_values, filter_shape);
    auto f_ind = filter_indices->matrix<int64>(); //[depth, height, width, output_channels, in_channels] //filters, channels, kernel_depth, kernel_height, kernel_width TODO: other cases?
    auto f_vals = filter_values->flat<T>();
    auto f_sh = filter_shape->flat<int64>();
    std::vector<int64> out_shape;
    sparseCuboidConv3D(in_ind, in_vals, in_sh, f_ind, f_vals, f_sh, stride_, output_map, out_shape);


    // Create an output tensor
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
    TensorShape out_ind_shape = {(int64) output_map.size(), (int64) in_ind.dimension(1)};
    TensorShape out_val_shape = {(int64) output_map.size()};
    TensorShape out_sh_shape = {(int64) in_ind.dimension(1)};
    OP_REQUIRES_OK(context, context->allocate_output("sparse_indices", out_ind_shape, &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("sparse_values", out_val_shape, &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("sparse_shape", out_sh_shape, &sparse_shape));

    auto out_ind = sparse_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto out_vals = sparse_values->flat<T>();
    auto out_sh = sparse_shape->flat<int64>();

    int64 idx = 0;
    for(auto it = output_map.begin(); it != output_map.end(); ++it, idx++){
        const std::vector<int64> &indice = it->first;
        for(int64 j = 0; j < indice.size(); ++j){
          out_ind(idx,j) = indice[j];
        }
        out_vals(idx) = it->second;
    }
    for(int64 idx = 0; idx < in_ind.dimension(1); ++idx){
        out_sh(idx) = out_shape[idx];
    }
    delete filter_indices, filter_values, filter_shape;
  }

  inline void
  dense_to_sparse(const Tensor* d_tensor, Tensor* s_indices, Tensor* s_values, Tensor* s_shape, const T epsilon = 0) const {
    std::map<std::vector<int64>, T> sparse_map;
    int64 dims = d_tensor->dims();

    auto d_flat = d_tensor->flat<T>();

    int64 non_zero_cnt = 0;
    for(int64 i = 0;  i < d_flat.size(); ++i){
      if(d_flat(i) > epsilon || d_flat(i) < epsilon) non_zero_cnt++;
    }

    delete(s_indices); delete(s_shape); delete(s_values);
    TensorShape out_ind_shape = {(int64) non_zero_cnt, dims};
    TensorShape out_val_shape = {(int64) non_zero_cnt};
    TensorShape out_sh_shape = {dims};
    s_indices = new Tensor(DT_INT64, out_ind_shape);
    s_shape = new Tensor(DT_INT64, out_sh_shape);
    s_values = new Tensor(d_tensor->dtype(), out_sh_shape);
    auto out_sh = s_shape->flat<int64>();
    for(size_t i = 0; i < out_sh.size(); ++i){
      out_sh(i) = d_tensor->dim_size(i);
    }


    switch (dims) {
#define NDIMS_CASE(N)                                                               \
  case N: {                                                                         \
    Converter<T, N> c;                                                              \
    c.convert(d_tensor, s_indices, s_values, epsilon);                              \
  } break;

      NDIMS_CASE(4);
      NDIMS_CASE(5);
#undef NDIMS_CASE
    }
  }

 private:
  std::vector<int32> stride_;
};




#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorDenseConv3D").Device(DEVICE_CPU), SparseTensorDenseConv3D<CPUDevice, type>);

REGISTER_CPU(float);
//REGISTER_CPU(double);
//REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);
