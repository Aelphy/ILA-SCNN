#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


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
REGISTER_OP("SparseTensorSparseKernelDenseConv3D")
  .Attr("T: {int32}")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Output("sparse_indices: int64")
  .Output("sparse_values: T")
  .Output("sparse_shape: int64")
  .Attr("strides: list(int) >= 5");
//  .Output("debug_output: string")





#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T>
class SparseTensorSparseKernelDenseConv3D : public OpKernel {
 public:
  explicit SparseTensorSparseKernelDenseConv3D(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
    OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
    OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
    auto in_ind = in_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    auto f_ind = filter_indices->matrix<int64>(); //filters, channels, kernel_depth, kernel_height, kernel_width TODO: other cases?
    auto f_vals = filter_values->flat<T>();
    auto f_sh = filter_shape->flat<int64>();

    std::map<std::vector<int64>, T> output_map; //stores the values for the output tensor

    //preparation: find center of filter
    std::vector<int64> filter_offset(f_ind.dimension(1));
    for(int64 i = 0; i < filter_offset.size(); ++i){
      filter_offset[i] = (f_sh(i) - 1) / 2; //TODO: precompute filter indices with offset? 
    }

    for(int64 i = 0; i < in_ind.dimension(0); ++i){ //TODO: parallelize filtering
      if(!index_is_unaffected_by_stride(in_ind, stride_, i)) continue; // go through all indices of input and check if valid (not affected by stride)

      //a) prepare filter to update output based on current value
      std::map<std::vector<int64>, T> filter_update;
      std::vector<int64> iids(in_ind.dimension(1));
      for(int64 j = 0; j < in_ind.dimension(1); j++){
        iids[j] = in_ind(i,j);
      }
      auto &input_value = in_vals(i);
      for(int64 j = 0; j < f_ind.dimension(0); j++){
        if(f_ind(j, 1) != in_ind(i,0)) continue; //filter channel != input channel
        bool is_in_bound = true;
        std::vector<int64> update_ids(in_ind.dimension(1), 0);
        assert(update_ids.size() >= f_ind.dimension(1) - 1);
        update_ids[0] = f_ind(j,0); //output channel is filter number
        for(int64 k = 2; k < f_ind.dimension(1); k++){ //TODO: ugly coding style... prototype
          update_ids[k - 1] = iids[k - 1] + f_ind(j,k) - filter_offset[k];  //depth, width and height
          if(update_ids[k - 1] < 0 || update_ids[k - 1] >= in_sh(k-1)){ //check boundaries
            is_in_bound = false;
            break;
          }
        }
        if(is_in_bound){
          T update_val = input_value * f_vals(j); //input value times filter weight at index
          filter_update.insert(std::make_pair(update_ids, update_val));
        }
      }
      

      //b) check if output exists (search required) and create or update output based on filter_update
        //TODO: concept: tree search to find upper and lower filter bound; binary search in between?
      for(auto it = filter_update.begin(); it != filter_update.end(); ++it){
        auto res_it = output_map.find(it->first);
        if(res_it == output_map.end()){
          output_map.insert(*it);
        } else {
          res_it->second += it->second;
        }
      }
    }

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
        if(idx == 0){
          out_sh(idx) = f_sh(idx); //output number of channels == number of filters
        } else {
          out_sh(idx) = in_sh(idx);
        }
    }


    //debug output
/*    Tensor* debug_ = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &debug_));
    auto output_ = debug_->scalar<string>();
    //std::stringstream debug; debug << "DEBUG OUTPUT:" << std::endl; 
    //std::string deb_string = debug.str();
    output_() = "test";
*/  
  }

 private:
  template<typename Matrix2DT>  
  inline bool index_is_unaffected_by_stride(const Matrix2DT& ids, const std::vector<int32>& stride, const int& a_id) const {
    //assert(id.size() == stride.size());
    for(int32 i = 0; i < stride.size(); ++i){
      if(stride[i] > 1){
        if(((ids(a_id,i) + 1) % stride[i]) == 0){
          return false;
        }
      }
    }
    return true;
  }

  std::vector<int32> stride_;
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConv3D").Device(DEVICE_CPU), SparseTensorSparseKernelDenseConv3D<CPUDevice, type>);

//REGISTER_CPU(float);
//REGISTER_CPU(double);
REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);

