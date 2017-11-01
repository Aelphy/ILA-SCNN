#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


REGISTER_OP("EvaluateConfusionMatrix")
  .Attr("T: realnumbertype")
  .Input("confusion_matrix: T")
  .Output("iou: float")
  .Output("average_iou: float")
  .Output("overall_accuracy: float");


#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class EvaluateConfusionMatrix : public OpKernel {
 public:
  explicit EvaluateConfusionMatrix(OpKernelConstruction* context) : OpKernel(context) 
  {}

  void Compute(OpKernelContext* context) override {
    const Tensor *confusion_matrix;
    OP_REQUIRES_OK(context, context->input("confusion_matrix", &confusion_matrix));
    auto conf = confusion_matrix->matrix<T>(); //channels, depth, height, width, optionally others TODO: other cases?

    // Create an output tensor
    Tensor *iou = NULL, *average_iou = NULL, *overall_accuracy = NULL;
    TensorShape iou_shape = {conf.dimension(0)};
    TensorShape average_iou_shape = {1};
    TensorShape overall_accuracy_shape = {1};
    OP_REQUIRES_OK(context, context->allocate_output("iou", iou_shape, &iou));
    OP_REQUIRES_OK(context, context->allocate_output("average_iou", average_iou_shape, &average_iou));
    OP_REQUIRES_OK(context, context->allocate_output("overall_accuracy", overall_accuracy_shape, &overall_accuracy));

    auto iou_ = iou->flat<float>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto aiou_ = average_iou->flat<float>();
    auto oa_ = overall_accuracy->flat<float>();
    std::vector<float> sum_x(conf.dimension(0), 0);
    std::vector<float> sum_y(conf.dimension(1), 0);
    float trace = 0;
    float sum_all = 0;
    for(size_t  i = 0; i < conf.dimension(0); ++i){
      trace += conf(i,i);
      for(size_t j = 0; j < conf.dimension(1); ++j){
        sum_x[i] += conf(i,j);
        sum_y[j] += conf(i,j);
        sum_all += conf(i,j);
      }
    }

    float all_iou = 0;
    for(size_t i = 0; i < conf.dimension(0); ++i){
      iou_(i) = conf(i,i) / (sum_x[i] + sum_y[i] - conf(i,i));
      all_iou += iou_(i);
    }

    aiou_(0) = all_iou / conf.dimension(0);     
    oa_(0) = trace / sum_all;
  }
};


#define REGISTER_CPU_ALL(type)                                                               \
  REGISTER_KERNEL_BUILDER(Name("EvaluateConfusionMatrix").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
                            EvaluateConfusionMatrix<CPUDevice, type>); 

  REGISTER_CPU_ALL(float);
  REGISTER_CPU_ALL(double);
  REGISTER_CPU_ALL(int32);
  //REGISTER_CPU_ALL(complex64);
  //REGISTER_CPU_ALL(complex128);
#undef REGISTER_CPU_ALL

