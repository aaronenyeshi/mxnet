/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_activation-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_CUDNN_ACTIVATION_INL_H_
#define MXNET_OPERATOR_CUDNN_ACTIVATION_INL_H_
#include <algorithm>
#include <vector>
#include "./activation-inl.h"

namespace mxnet {
namespace op {
template<typename DType>
class CuDNNActivationOp : public Operator {
 public:
  explicit CuDNNActivationOp(ActivationParam param) {
    param_ = param;
    init_cudnn_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;
    switch (param_.act_type) {
      case activation::kReLU:
        mode_ = miopenActivationRELU;
        break;
      case activation::kSigmoid:
        mode_ = miopenActivationLOGISTIC;
        break;
      case activation::kTanh:
        mode_ = miopenActivationTANH;
        break;
      default:
        LOG(FATAL) << "Not implmented";
        break;
    }
    #if CUDNN_MAJOR >= 5
    //nan_prop_ = CUDNN_NOT_PROPAGATE_NAN; //TODO not supported
    CUDNN_CALL(miopenCreateActivationDescriptor(&desc_));
    double alpha = 1.0f; //TODO temporary fix for arguments
    double beta  = 0.0f; //TODO temporary fix for arguments
    CUDNN_CALL(miopenSetActivationDescriptor(desc_, mode_, alpha , beta ,relu_ceil_)); //TODO temporary fix
    #endif
  }

  ~CuDNNActivationOp() {
    if (init_cudnn_) {
      CUDNN_CALL(miopenDestroyTensorDescriptor(shape_desc_));
      #if CUDNN_MAJOR >= 5
      CUDNN_CALL(miopenDestroyActivationDescriptor(desc_));
      #endif
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> data;
    Tensor<gpu, 4, DType> out;
    if (in_data[activation::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[activation::kData].shape_[0],
                               in_data[activation::kData].shape_[1], 1, 1);
      data = in_data[activation::kData].get_with_shape<gpu, 4, DType>(dshape, s);
      out = out_data[activation::kOut].get_with_shape<gpu, 4, DType>(dshape, s);
    } else {
      Shape<4> dshape;
      index_t size_left = in_data[activation::kData].Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_data[activation::kData].ndim()) {
          dshape[i] = in_data[activation::kData].shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      data = in_data[activation::kData].get_with_shape<gpu, 4, DType>(dshape, s);
      out = out_data[activation::kOut].get_with_shape<gpu, 4, DType>(dshape, s);
    }
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      CUDNN_CALL(miopenCreateTensorDescriptor(&shape_desc_));
      CUDNN_CALL(miopenSet4dTensorDescriptor(shape_desc_,
                                            dtype_,
                                            data.shape_[0],
                                            data.shape_[1],
                                            data.shape_[2],
                                            data.shape_[3]));
    }
//Miopen supports API as per cudnn_version 5, API porting has been done as per version 5 and removed cudnn_version 4 code
/*    #if CUDNN_MAJOR <= 4
    CUDNN_CALL(miopenActivationForward(s->dnn_handle_,
                                      mode_,
                                      &alpha,
                                      shape_desc_,
                                      data.dptr_,
                                      &beta,
                                      shape_desc_,
                                      out.dptr_));
    #elif CUDNN_MAJOR >= 5*/
    CUDNN_CALL(miopenActivationForward(s->dnn_handle_,
                                     desc_,
                                     &alpha,
                                     shape_desc_,
                                     data.dptr_,
                                     &beta,
                                     shape_desc_,
                                     out.dptr_));
   // #endif
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    typename DataType<DType>::ScaleType alpha = 1.0f;
    typename DataType<DType>::ScaleType beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> grad;
    Tensor<gpu, 4, DType> data;
    Tensor<gpu, 4, DType> output_data;
    Tensor<gpu, 4, DType> input_grad;
    if (in_grad[activation::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_grad[activation::kData].shape_[0],
                               in_grad[activation::kData].shape_[1], 1, 1);
      data = in_data[activation::kData].get_with_shape<gpu, 4, DType>(dshape, s);
      grad = out_grad[activation::kOut].get_with_shape<gpu, 4, DType>(dshape, s);
      output_data = out_data[activation::kOut].get_with_shape<gpu, 4, DType>(dshape, s);
      input_grad = in_grad[activation::kData].get_with_shape<gpu, 4, DType>(dshape, s);
    } else {
      Shape<4> dshape;
      index_t size_left = in_grad[activation::kData].Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_grad[activation::kData].ndim()) {
          dshape[i] = in_grad[activation::kData].shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      data = in_data[activation::kData].get_with_shape<gpu, 4, DType>(dshape, s);
      output_data = out_data[activation::kOut].get_with_shape<gpu, 4, DType>(dshape, s);
      grad = out_grad[activation::kOut].get_with_shape<gpu, 4, DType>(dshape, s);
      input_grad = in_grad[activation::kData].get_with_shape<gpu, 4, DType>(dshape, s);
    }
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    //Miopen supports API as per cudnn_version 5, API porting has been done as per version 5 and removed cudnn_version 4 code
    /*#if CUDNN_MAJOR <= 4
    CUDNN_CALL(miopenActivationBackward(s->dnn_handle_,
                                       mode_,
                                       &alpha,
                                       shape_desc_,
                                       output_data.dptr_,
                                       shape_desc_,
                                       grad.dptr_,
                                       shape_desc_,
                                       data.dptr_,
                                       &beta,
                                       shape_desc_,
                                       input_grad.dptr_));
    #elif CUDNN_MAJOR >= 5*/
    CUDNN_CALL(miopenActivationBackward(s->dnn_handle_,
                                       desc_,
                                       &alpha,
                                       shape_desc_,
                                       output_data.dptr_,
                                       shape_desc_,
                                       grad.dptr_,
                                       shape_desc_,
                                       data.dptr_,
                                       &beta,
                                       shape_desc_,
                                       input_grad.dptr_));
//    #endif
  }

 private:
  bool init_cudnn_;
  miopenDataType_t dtype_;
  miopenActivationMode_t mode_;
  miopenTensorDescriptor_t shape_desc_;
  ActivationParam param_;
//#if CUDNN_MAJOR >= 5

  miopenActivationDescriptor_t desc_;
  //cudnnNanPropagation_t nan_prop_; //TODO not supported
  double relu_ceil_;
//#endif
};  // class CuDNNActivationOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUDNN_ACTIVATION_INL_H_
