/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_activation-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_CUDNN_SOFTMAX_ACTIVATION_INL_H_
#define MXNET_OPERATOR_CUDNN_SOFTMAX_ACTIVATION_INL_H_
#include <algorithm>
#include <vector>
#include "./softmax_activation-inl.h"

namespace mxnet {
namespace op {
class CuDNNSoftmaxActivationOp : public Operator {
 public:
  explicit CuDNNSoftmaxActivationOp(SoftmaxActivationParam param) {
    this->param_ = param;
    init_cudnn_ = false;
    dtype_ = miopenFloat;
  }

  ~CuDNNSoftmaxActivationOp() {
    if (init_cudnn_) {
      CUDNN_CALL(miopenDestroyTensorDescriptor(shape_desc_));
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
    Tensor<gpu, 4> data;
    Tensor<gpu, 4> out;
    //cudnnSoftmaxMode_t softmax_mode; //MIOpen does not support Softmax modes.
    if (param_.mode == softmax_activation::kInstance) {
      CHECK_EQ(in_data[softmax_activation::kData].ndim(), 2)
        << "Input need to have 2 dimensions when mode=instance.";
      Shape<4> dshape = Shape4(in_data[softmax_activation::kData].shape_[0],
                               in_data[softmax_activation::kData].shape_[1], 1, 1);
      data = in_data[softmax_activation::kData].get_with_shape<gpu, 4, real_t>(dshape, s);
      out = out_data[softmax_activation::kOut].get_with_shape<gpu, 4, real_t>(dshape, s);
      //softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE; //MIOpen does not support Softmax modes.
    } else {
      CHECK_GE(in_data[softmax_activation::kData].ndim(), 3)
        << "Input need to have a least 3 dimensions when mode=channel";
      Shape<4> dshape;
      index_t size_left = in_data[softmax_activation::kData].Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_data[softmax_activation::kData].ndim()) {
          dshape[i] = in_data[softmax_activation::kData].shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      data = in_data[softmax_activation::kData].get_with_shape<gpu, 4, real_t>(dshape, s);
      out = out_data[softmax_activation::kOut].get_with_shape<gpu, 4, real_t>(dshape, s);
      //softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL; //MIOpen does not support Softmax modes.
    }
    float alpha = 1.0f;
    float beta = 0.0f;
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
    CUDNN_CALL(miopenSoftmaxForward(s->dnn_handle_,
                                   &alpha,
                                   shape_desc_,
                                   data.dptr_,
                                   &beta,
                                   shape_desc_,
                                   out.dptr_));
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
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    float alpha = 1.0f;
    float beta = 0.0f;
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4> grad;
    Tensor<gpu, 4> data;
    Tensor<gpu, 4> output_data;
    Tensor<gpu, 4> input_grad;
    //cudnnSoftmaxMode_t softmax_mode; //MIOpen does not support Softmax modes.
    if (param_.mode == softmax_activation::kInstance) {
      CHECK_EQ(in_grad[softmax_activation::kData].ndim(), 2)
        << "Input need to have 2 dimensions when mode=instance.";
      Shape<4> dshape = Shape4(in_grad[softmax_activation::kData].shape_[0],
                               in_grad[softmax_activation::kData].shape_[1], 1, 1);
      grad = out_grad[softmax_activation::kOut].get_with_shape<gpu, 4, real_t>(dshape, s);
      output_data = out_data[softmax_activation::kOut].get_with_shape<gpu, 4, real_t>(dshape, s);
      input_grad = in_grad[softmax_activation::kData].get_with_shape<gpu, 4, real_t>(dshape, s);
      //softmax_mode = CUDNN_SOFTMAX_MODE_INSTANCE; //MIOpen does not support Softmax modes.
    } else {
      CHECK_GE(in_grad[softmax_activation::kData].ndim(), 3)
        << "Input need to have a least 3 dimensions when mode=channel";
      Shape<4> dshape;
      index_t size_left = in_grad[softmax_activation::kData].Size();
      for (int i = 0; i < 3; ++i) {
        if (i < in_grad[softmax_activation::kData].ndim()) {
          dshape[i] = in_grad[softmax_activation::kData].shape_[i];
        } else {
          dshape[i] = 1;
        }
        size_left /= dshape[i];
      }
      dshape[3] = size_left;
      output_data = out_data[softmax_activation::kOut].get_with_shape<gpu, 4, real_t>(dshape, s);
      grad = out_grad[softmax_activation::kOut].get_with_shape<gpu, 4, real_t>(dshape, s);
      input_grad = in_grad[softmax_activation::kData].get_with_shape<gpu, 4, real_t>(dshape, s);
      //softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL; //MIOpen does not support Softmax modes.
    }
    CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
    CUDNN_CALL(miopenSoftmaxBackward(s->dnn_handle_,
                                    &alpha,
                                    shape_desc_,
                                    output_data.dptr_,
                                    shape_desc_,
                                    grad.dptr_,
                                    &beta,
                                    shape_desc_,
                                    input_grad.dptr_));
  }

 private:
  bool init_cudnn_;
  miopenDataType_t dtype_;
  miopenTensorDescriptor_t  shape_desc_;
  SoftmaxActivationParam param_;
};  // class CuDNNSoftmaxActivationOp
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CUDNN_SOFTMAX_ACTIVATION_INL_H_
