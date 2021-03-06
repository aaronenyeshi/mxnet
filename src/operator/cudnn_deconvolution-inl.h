/*!
 * Copyright (c) 2017 by Contributors
 * \file cudnn_deconvolution-inl.h
 * \brief
 * \author Wei Wu, Leonard Lausen
*/
#ifndef MXNET_OPERATOR_CUDNN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_CUDNN_DECONVOLUTION_INL_H_

#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include "./deconvolution-inl.h"
#include "./cudnn_algoreg-inl.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

template<typename DType>
class CuDNNDeconvolutionOp : public Operator {
 public:
  explicit CuDNNDeconvolutionOp(DeconvolutionParam param,
                                int forward_compute_type,
                                int backward_compute_type,
                                const std::vector<TShape>& in_shape,
                                const std::vector<TShape>& out_shape,
                                const Context& ctx) {
    using namespace mshadow;
    this->param_ = param;
    auto cudnn_forward_compute_type = convertToCuDNNDataType(forward_compute_type);
    auto cudnn_backward_compute_type = convertToCuDNNDataType(backward_compute_type);
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    init_cudnn_ = false;
    init_temp_size_ = false;
    dtype_ = mshadow::DataType<DType>::kCudnnFlag;

#if CUDNN_MAJOR >= 5
    MSHADOW_LAYOUT_SWITCH(param_.layout.value(), Layout, {
       // format_ = LayoutType<Layout>::kCudnnFlag;
      });
#else
    CHECK(param_.layout.value() == kNCHW || param_.layout.value() == kNCDHW)
      << "Need CuDNN > 5.0 for layout support";
#endif
    // Double check to make sure this class supports the operation
    if (!Supports(param, forward_compute_type, backward_compute_type))
      LOG(FATAL) << "Need CuDNN >= 6.0 for dilated convolution.";

    InitDescriptors(ctx, in_shape, out_shape,
                    cudnn_forward_compute_type, cudnn_backward_compute_type);

    if (!param_.cudnn_tune) {
      param_.cudnn_tune = dmlc::GetEnv("MXNET_CUDNN_AUTOTUNE_DEFAULT", 1);
    }
    // In cuDNN_v6, dilated convolution descriptors are compatible with only a
    // single convolution algorithm.  Despite this, we go through the algorithm
    // selection process, which will return the only algorithm supported.  This
    // approach keeps the treatment of convolution cases uniform and will
    // naturally respond to more algorithms supporting dilated convolutions in
    // future cuDNN releases.
    SelectAlgo(ctx, in_shape, out_shape,
               cudnn_forward_compute_type, cudnn_backward_compute_type);
  }

  ~CuDNNDeconvolutionOp() {
    if (init_cudnn_) {
      CUDNN_CALL(miopenDestroyTensorDescriptor(in_desc_));
      CUDNN_CALL(miopenDestroyTensorDescriptor(out_desc_));
      CUDNN_CALL(miopenDestroyTensorDescriptor(bias_desc_));
      CUDNN_CALL(miopenDestroyTensorDescriptor(filter_desc_));
      CUDNN_CALL(miopenDestroyConvolutionDescriptor(forward_conv_desc_));
      CUDNN_CALL(miopenDestroyConvolutionDescriptor(backward_conv_desc_));
    }
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = param_.no_bias ? 2 : 3;
    DType *data_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *out_ptr = NULL;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    GetTempSize(ctx);
    Tensor<gpu, 1, DType> workspace =
      ctx.requested[deconv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(forward_workspace_), s);

    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> data = in_data[deconv::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[deconv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> out = out_data[deconv::kOut].get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    } else {
      Tensor<gpu, 5, DType> data = in_data[deconv::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> wmat = in_data[deconv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> out = out_data[deconv::kOut].get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      CHECK_EQ(wmat.CheckContiguous(), true);
      CHECK_EQ(out.CheckContiguous(), true);
      data_ptr = data.dptr_;
      wmat_ptr = wmat.dptr_;
      out_ptr = out.dptr_;
    }

    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType alpha2 = 1.0f; //set to zero as per CUDNN and MIOpen documentation
      typename DataType<DType>::ScaleType beta  = 0.0f;


      //#if CUDNN_MAJOR <= 4  // removed if condition as there is no equivalent to cudnnConvolutionBackwardData_v3 in miopen need to recheck


// cudnnConvolutionBackwardData_v3 is equivalent to cudnnConvolutionBackwardData as per cuDNN library                                             documentation for ref:  http://snurran.sics.se/hops/cudnn/v4-cudnn_library.pdf
     /* CUDNN_CALL(cudnnConvolutionBackwardData_v3(s->dnn_handle_,
                 &alpha,
                 filter_desc_,
                 wmat_ptr + weight_offset_ * g,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 forward_conv_desc_,  // this backward algorithm used for inference
                 back_algo_,
                 workspace.dptr_,
                 backward_workspace_byte_,
                 &beta,
                 out_desc_,
                 out.dptr_ + out_offset_ * g));*/ //Not supported
//      #elif CUDNN_MAJOR >= 5

      CUDNN_CALL(miopenConvolutionBackwardData(s->dnn_handle_,
                 &alpha,
                 in_desc_,
                 data_ptr + data_offset_ * g,
                 filter_desc_,
                 wmat_ptr + weight_offset_ * g,
                 forward_conv_desc_,
                 back_algo_,
                 &beta,
                 out_desc_,
                 out_ptr + out_offset_ * g,
                 workspace.dptr_,
                 backward_workspace_byte_));
  //    #endif
      if (!param_.no_bias) {
        // beta = 1.0f;
        Tensor<gpu, 1, DType> bias = in_data[deconv::kBias].get<gpu, 1, DType>(s);
//#if CUDNN_MAJOR >= 4
#if CUDNN_MAJOR>=3 //TODO changed hip condition as there is no support for cudnnAddTensor with CUDNN_ADD_SAME_C option.
        // changed parameters order as per the documented functionality of the API
        CUDNN_CALL(miopenOpTensor(s->dnn_handle_,
                                  miopenTensorOpAdd,
				  &alpha,
                                  bias_desc_,
                                  bias.dptr_ + bias_offset_ * g,
                                  &alpha2,
                                  out_desc_,
                                  out_ptr + out_offset_ * g,
                                  &beta,
                                  out_desc_,
                                  out_ptr + out_offset_ * g));
#endif
/*#if CUDNN_MAJOR == 3
      //changed parameters order as per the documented functionality of the API
        CUDNN_CALL(miopenOpTensor(s->dnn_handle_,
                                  miopenTensorOpAdd,
                                  &alpha,
                                  bias_desc_,
                                  bias.dptr_ + bias_offset_ * g,
                                  &alpha2,
                                  out_desc_,
                                  out_ptr + out_offset_ * g,
                                  &beta,
                                  out_desc_,
                                  out_ptr + out_offset_ * g));
#endif*/ //TODO removed and added in above condition.
      }
    }
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
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    DType *grad_ptr = NULL;
    DType *wmat_ptr = NULL;
    DType *gwmat_ptr = NULL;
    DType *data_ptr = NULL;
    DType *gdata_ptr = NULL;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    if (param_.kernel.ndim() == 2) {
      Tensor<gpu, 4, DType> grad = out_grad[deconv::kOut].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> wmat = in_data[deconv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gwmat = in_grad[deconv::kWeight].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> data = in_data[deconv::kData].get<gpu, 4, DType>(s);
      Tensor<gpu, 4, DType> gdata = in_grad[deconv::kData].get<gpu, 4, DType>(s);
      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = gdata.dptr_;
    } else {
      Tensor<gpu, 5, DType> grad = out_grad[deconv::kOut].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> wmat = in_data[deconv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> gwmat = in_grad[deconv::kWeight].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> data = in_data[deconv::kData].get<gpu, 5, DType>(s);
      Tensor<gpu, 5, DType> gdata = in_grad[deconv::kData].get<gpu, 5, DType>(s);
      grad_ptr = grad.dptr_;
      wmat_ptr = wmat.dptr_;
      gwmat_ptr = gwmat.dptr_;
      data_ptr = data.dptr_;
      gdata_ptr = gdata.dptr_;
    }
    CHECK_NE(req[deconv::kWeight], kWriteInplace);
    CHECK_NE(req[deconv::kBias], kWriteInplace);
    CHECK_NE(req[deconv::kData], kWriteInplace);
    Tensor<gpu, 1, DType> workspace =
        ctx.requested[deconv::kTempSpace].get_space_typed<gpu, 1, DType>(
                                 mshadow::Shape1(backward_workspace_), s);
    for (uint32_t g = 0; g < param_.num_group; ++g) {
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType bias_beta =
        req[deconv::kBias] == kAddTo ? 1.0f : 0.0f;
      typename DataType<DType>::ScaleType data_beta =
        req[deconv::kData] == kAddTo ? 1.0f : 0.0f;
      typename DataType<DType>::ScaleType weight_beta =
        req[deconv::kWeight] == kAddTo ? 1.0f : 0.0f;
      if (!param_.no_bias && (req[deconv::kBias] != kNullOp)) {
        Tensor<gpu, 1, DType> gbias = in_grad[deconv::kBias].get<gpu, 1, DType>(s);
        CUDNN_CALL(miopenConvolutionBackwardBias(s->dnn_handle_,
                                                &alpha,
                                                out_desc_,
                                                grad_ptr + out_offset_ * g,
                                                &bias_beta,
                                                bias_desc_,
                                                gbias.dptr_ + bias_offset_ * g));
      }
      if (req[deconv::kWeight] != kNullOp) {
//        #if CUDNN_MAJOR <= 4 //removed if condition as there is no equivalent to cudnnConvolutionBackwardData_v3 in miopen need to recheck
// cudnnConvolutionBackwardData_v3 is equivalent to cudnnConvolutionBackwardData as per cuDNN library                                             documentation for ref:  http://snurran.sics.se/hops/cudnn/v4-cudnn_library.pdf

        /*CUDNN_CALL(cudnnConvolutionBackwardFilter_v3(
          s->dnn_handle_,
          &alpha,
          out_desc_,
          grad_ptr + out_offset_ * g,
          in_desc_,
          data_ptr + data_offset_ * g,
          backward_conv_desc_,
          back_algo_w_,
          workspace.dptr_,
          backward_workspace_byte_,
          &weight_beta,
          filter_desc_,
          gwmat.dptr_ + weight_offset_ * g));*/ //Not supported
  //      #elif CUDNN_MAJOR >= 5
        CUDNN_CALL(miopenConvolutionBackwardWeights(
          s->dnn_handle_,
          &alpha,
          in_desc_,
          data_ptr + data_offset_ * g,
          out_desc_,
          grad_ptr + out_offset_ * g,
          backward_conv_desc_,
          back_algo_w_,
          &weight_beta,
          filter_desc_,
          gwmat_ptr + weight_offset_ * g,
          workspace.dptr_,
          backward_workspace_byte_));
    //    #endif
      }
      if (req[deconv::kData] != kNullOp) {
        CUDNN_CALL(miopenConvolutionForward(s->dnn_handle_,
                                           &alpha,
                                           out_desc_,
                                           grad_ptr + out_offset_ * g,
                                           filter_desc_,
                                           wmat_ptr + weight_offset_ * g,
                                           backward_conv_desc_,
                                           algo_,
                                           &data_beta,
                                           in_desc_,
                                           gdata_ptr + data_offset_ * g,
                                           workspace.dptr_,
                                           forward_workspace_byte_));
      }
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the deconvolution
 * operation described by `param`: cuDNN v5 and earlier does not support
 * dilated convolutions.
 */
  static bool Supports(DeconvolutionParam param,
                       int forward_compute_type,
                       int backward_compute_type) {
    using namespace mshadow;

    // NDHWC not supported, NHWC not supported in true fp16
    auto layout_val = param.layout.value();
    auto true_fp16 = DataType<DType>::kFlag == kFloat16 &&
      (forward_compute_type == kFloat16 || backward_compute_type == kFloat16);
    if (layout_val == kNDHWC || layout_val == kNHWC && true_fp16)
      return false;

    // The factor by which the effective filter size grows based on dilation.
    auto filterDilationFactor = param.dilate.Size();

    // The v6 kernels that backprop a dilated convolution don't handle fp16.
    // Since the deconvolution "forward" kernel is really a backprop-to-data
    // cuDNN kernel, the following logic is slightly different than that
    // used in CuDNNConvolution::Supports().

    // Dilation support across all architectures only available after v6.0.20.
    return filterDilationFactor == 1 ||
           filterDilationFactor > 1 && (CUDNN_VERSION > 6020) &&
           (backward_compute_type != kFloat16) &&
           (forward_compute_type != kFloat16);
  }

 private:
/*!
 * \brief Translate an mxnet datatype to the corresponding miopenDataType_t.
 */
  miopenDataType_t convertToCuDNNDataType(int dtype) {
    miopenDataType_t converted = miopenFloat;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_REAL_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

  inline void InitDescriptors(const Context& ctx,
                              const std::vector<TShape> &in_shape,
                              const std::vector<TShape> &out_shape,
                              miopenDataType_t cudnn_forward_compute_type,
                              miopenDataType_t cudnn_backward_compute_type) {
    using namespace mshadow;
    #if CUDNN_MAJOR >= 5
    //format_ = CUDNN_TENSOR_NCHW;
    #endif
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);
    CUDNN_CALL(miopenCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&bias_desc_));
    CUDNN_CALL(miopenCreateTensorDescriptor(&filter_desc_));
    CUDNN_CALL(miopenCreateConvolutionDescriptor(&forward_conv_desc_));
    CUDNN_CALL(miopenCreateConvolutionDescriptor(&backward_conv_desc_));

    TShape dshape = in_shape[deconv::kData];
    TShape wshape = in_shape[deconv::kWeight];
    TShape oshape = out_shape[deconv::kOut];
    TShape dstride, ostride, wstride;
    wshape[0] /= param_.num_group;

    if (param_.kernel.ndim() == 2) {
      // 2d conv
      index_t o_pad[2];
      index_t o_adj[2];
      param_.InferPad(dshape, o_pad, o_adj);



//New parameter 'computeType' which is added from cuDNN version 6 is not supported in MIOpen so commented the piece of code under version 6.
     //Order of parameters changed as per documentation
      /*#if CUDNN_MAJOR >= 6
      CUDNN_CALL(miopenInitConvolutionDescriptor(forward_conv_desc_,
                                                 miopenTranspose,//CUDNN_CROSS_CORRELATION,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 param_.stride[0],
                                                 param_.stride[1],
                                                 param_.dilate[1],
                                                 param_.dilate[0]));
      CUDNN_CALL(miopenInitConvolutionDescriptor(backward_conv_desc_,
                                                 miopenTranspose,//CUDNN_CROSS_CORRELATION,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 param_.stride[0],
                                                 param_.stride[1],
                                                 param_.dilate[1],
                                                 param_.dilate[0]));
      #else*/
      CUDNN_CALL(miopenInitConvolutionDescriptor(forward_conv_desc_,
                                                 miopenTranspose,//CUDNN_CROSS_CORRELATION,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 param_.stride[0],
                                                 param_.stride[1],
                                                 param_.dilate[1],
                                                 param_.dilate[0]));
      CUDNN_CALL(miopenInitConvolutionDescriptor(backward_conv_desc_,
                                                 miopenTranspose,//CUDNN_CROSS_CORRELATION,
                                                 o_pad[0],
                                                 o_pad[1],
                                                 param_.stride[0],
                                                 param_.stride[1],
                                                 param_.dilate[1],
                                                 param_.dilate[0]));
     // #endif

      #if CUDNN_MAJOR >= 5
      wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
      CUDNN_CALL(miopenSet4dTensorDescriptor(filter_desc_,
                                            dtype_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));
      #else
      CHECK_EQ(param_.layout.value(), kNCHW) << "CuDNN V4 only support NCHW layout";
      CUDNN_CALL(miopenSet4dTensorDescriptor(filter_desc_,
                                            dtype_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));
      #endif
      wstride = ConvertLayout(Shape4(wshape[1] * wshape[2] * wshape[3],
                                     wshape[2] * wshape[3],
                                     wshape[3],
                                     1),
                             param_.layout.value(), kNCHW);

      dstride = ConvertLayout(Shape4(dshape[1] * dshape[2] * dshape[3],
                                     dshape[2] * dshape[3],
                                     dshape[3],
                                     1),
                              param_.layout.value(), kNCHW);
      dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);

      ostride = ConvertLayout(Shape4(oshape[1] * oshape[2] * oshape[3],
                                     oshape[2] * oshape[3],
                                     oshape[3],
                                     1),
                              param_.layout.value(), kNCHW);
      oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
    } else if (param_.kernel.ndim() == 3) {
      // 3d conv
      index_t o_pad[3];
      index_t o_adj[3];
      param_.InferPad(dshape, o_pad, o_adj);

      #if CUDNN_MAJOR >= 5
      CHECK_EQ(param_.layout.value(), kNCDHW) << "CuDNN only support 3D conv with NCDHW layout";


      int dimensn = static_cast<int>(wshape.ndim());
    if (dimensn > 0 && dimensn < 6)
      CUDNN_CALL(miopenSetTensorDescriptor(filter_desc_,
                                        dtype_,
                                        static_cast<int>(wshape.ndim()),
                                        reinterpret_cast<int*>(&wshape[0]),
                                        reinterpret_cast<int*>(&wstride[0])));
      #else
      LOG(FATAL) << "Only support CUDNN V5 for 3D convolution";
      #endif
     //TODO : need to recheck with miopenInitConvolutionDescriptor() or for relavent API in miopen
     /*CUDNN_CALL(cudnnSetConvolutionNdDescriptor(forward_conv_desc_,
                                                 3,
                                                 reinterpret_cast<int*>(&o_pad[0]),
                                                 reinterpret_cast<int*>(&param_.stride[0]),
                                                 reinterpret_cast<int*>(&param_.dilate[0]),
                                                 miopenTranspose,//CUDNN_CROSS_CORRELATION,
                                                 cudnn_forward_compute_type));

      CUDNN_CALL(cudnnSetConvolutionNdDescriptor(backward_conv_desc_,
                                                 3,
                                                 reinterpret_cast<int*>(&o_pad[0]),
                                                 reinterpret_cast<int*>(&param_.stride[0]),
                                                 reinterpret_cast<int*>(&param_.dilate[0]),
                                                 miopenTranspose,//CUDNN_CROSS_CORRELATION,
                                                 cudnn_backward_compute_type));*/ //TODO . Unsupported
      dstride = ConvertLayout(Shape5(dshape[1] * dshape[2] * dshape[3] * dshape[4],
                                     dshape[2] * dshape[3] * dshape[4],
                                     dshape[3] * dshape[4],
                                     dshape[4],
                                     1),
                              param_.layout.value(), kNCDHW);
      dshape = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);

      ostride = ConvertLayout(Shape5(oshape[1] * oshape[2] * oshape[3] * oshape[4],
                                     oshape[2] * oshape[3] * oshape[4],
                                     oshape[3] * oshape[4],
                                     oshape[4],
                                     1),
                              param_.layout.value(), kNCDHW);
      oshape = ConvertLayout(oshape.get<5>(), param_.layout.value(), kNCDHW);
    }
    dshape[1] /= param_.num_group;
    oshape[1] /= param_.num_group;
    weight_offset_ = wshape.Size();
    data_offset_ = dstride[1] * dshape[1];
    out_offset_ = ostride[1] * oshape[1];
   // need to recheck Added if condition to allow dimensions between 1 to 5
    int dimensn = static_cast<int>(dshape.ndim());
    if (dimensn > 0 && dimensn < 6)
         CUDNN_CALL(miopenSetTensorDescriptor(in_desc_,
                                          dtype_, // miopenFloat is implemented
                                          static_cast<int>(dshape.ndim()),
                                          reinterpret_cast<int*>(&dshape[0]),
                                          reinterpret_cast<int*>(&dstride[0])));
    /*CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                          dtype_,
                                          static_cast<int>(dshape.ndim()),
                                          reinterpret_cast<int*>(&dshape[0]),
                                          reinterpret_cast<int*>(&dstride[0])));*/
// need to recheck Added if condition to allow dimensions between 1 to 5
    int ndim = static_cast<int>(oshape.ndim());
      if (ndim > 0 && ndim < 6)
          CUDNN_CALL(miopenSetTensorDescriptor(out_desc_,
                                          dtype_, //  miopenFloat is implemented
                                          static_cast<int>(oshape.ndim()),
                                          reinterpret_cast<int*>(&oshape[0]),
                                          reinterpret_cast<int*>(&ostride[0])));
    /* CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
                                          static_cast<int>(oshape.ndim()),
                                          reinterpret_cast<int*>(&oshape[0]),
                                          reinterpret_cast<int*>(&ostride[0])));*/

    if (!param_.no_bias) {
      TShape bias = in_shape[deconv::kBias];
      bias_offset_ = bias[0] / param_.num_group;
      std::vector<int> bias_shape = {1,
                                     static_cast<int>(bias[0] / param_.num_group),
                                     1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias_offset_), 1, 1, 1};
      if (param_.kernel.ndim() == 3) {
        bias_shape.push_back(1);
        bias_stride.push_back(1);
      }
   //need to recheck Added if condition to allow dimensions between 1 to 5
      ndim = static_cast<int>(bias_shape.size());
      if (ndim > 0 && ndim < 6)
             CUDNN_CALL(miopenSetTensorDescriptor(bias_desc_,
                                            dtype_, // miopenFloat is implemented
                                            static_cast<int>(bias_shape.size()),
                                            &bias_shape[0],
                                            &bias_stride[0]));
      /*CUDNN_CALL(cudnnSetTensorNdDescriptor(bias_desc_,
                                            dtype_,
                                            static_cast<int>(bias_shape.size()),
                                            &bias_shape[0],
                                            &bias_stride[0]));*/
    }
    init_cudnn_ = true;
  }

  void SelectAlgo(const Context& ctx,
                  const std::vector<TShape>& in_shape,
                  const std::vector<TShape>& out_shape,
                  miopenDataType_t cudnn_forward_compute_type,
                  miopenDataType_t cudnn_backward_compute_type) {
    std::string key = CuDNNAlgoReg::Get()->GetKey(param_, in_shape, out_shape, dtype_,
                                                  cudnn_forward_compute_type,
                                                  cudnn_backward_compute_type);
    if (CuDNNAlgoReg::Get()->Find(key, &algo_, &back_algo_, &back_algo_w_))
      return;

    Engine::VarHandle var = Engine::Get()->NewVariable();
    Engine::Get()->PushSync([=](RunContext rctx) {
      mshadow::Stream<gpu> *s = rctx.get_stream<gpu>();
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);
      size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));
      if (!param_.cudnn_tune.value()) {
        // In cuDNNv6, for kNHWC, only CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM is
        // supported.  Hard-coded this since the algo find() or get() throws an FPE.
        if (CUDNN_MAJOR == 6 && param_.layout.value() == mshadow::kNHWC) {
          algo_ = miopenConvolutionFwdAlgoGEMM;
        } else {
         /* CUDNN_CALL(miopenFindConvolutionForwardAlgorithm(s->dnn_handle_,
                     out_desc_,
                     filter_desc_,
                     backward_conv_desc_,  // forward algorithm used to backprop-to-data
                     in_desc_,
                     CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                     workspace_byte,
                     &(this->algo_)));*/ //TODO Temporarily commented.Parameters to be fixed
        }
        /*CUDNN_CALL(miopenFindConvolutionBackwardWeightsAlgorithm(s->dnn_handle_,
                   out_desc_,
                   in_desc_,
                   backward_conv_desc_,
                   filter_desc_,
                   CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                   workspace_byte,
                   &(this->back_algo_w_)));*/ //TODO Temporarily commented.Parameters to be fixed
        /*CUDNN_CALL(miopenFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                   filter_desc_,
                   in_desc_,
                   forward_conv_desc_,  // this backward algorithm used for inference
                   out_desc_,
                   CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                   workspace_byte,
                   &(this->back_algo_)));*/ //TODO Temporarily commented.Parameters to be fixed
      } else {
        const int kMaxAlgos = 10;
        int nalgo = kMaxAlgos;
        int i;

        // In cuDNNv6, for kNHWC, only CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM is
        // supported.  Hard-coded this since the algo find() or get() throws an FPE.
        if (CUDNN_MAJOR == 6 && param_.layout.value() == mshadow::kNHWC) {
          algo_ = miopenConvolutionFwdAlgoGEMM;
        } else {
          miopenConvAlgoPerf_t fwd_algo[kMaxAlgos];
         /* CUDNN_CALL(miopenFindConvolutionForwardAlgorithm(s->dnn_handle_,
                     out_desc_,
                     filter_desc_,
                     backward_conv_desc_,  // forward algorithm used to backprop-to-data
                     in_desc_,
                     kMaxAlgos,
                     &nalgo,
                     fwd_algo));*/ //TODO Temporarily commented.Parameters to be fixed
          i = 0;
          while (i < nalgo
               && (//fwd_algo[i].status != miopenStatusSuccess //TODO .Unsupported status variable in miopenConvAlgoPerf_t
                (param_.cudnn_tune.value() == deconv::kLimited
               && fwd_algo[i].memory > workspace_byte))) ++i;
          if (i == nalgo) {
            LOG(FATAL) << "Failed to find a 'forward' convolution algorithm " <<
              "(for use in deconvolution operator backprop-to-data).";
          } else {
              this->algo_ = fwd_algo[i].fwd_algo;
          }
        }

        miopenConvAlgoPerf_t bwd_filter_algo[kMaxAlgos];
        /*CUDNN_CALL(miopenFindConvolutionBackwardWeightsAlgorithm(s->dnn_handle_,
                   out_desc_,
                   in_desc_,
                   backward_conv_desc_,
                   filter_desc_,
                   kMaxAlgos,
                   &nalgo,
                   bwd_filter_algo));*/ //TODO Temporarily commented.Parameters to be fixed
        i = 0;
        while (i < nalgo
               && (//bwd_filter_algo[i].status != miopenStatusSuccess //TODO .Unsupported status variable in miopenConvAlgoPerf_t
                (param_.cudnn_tune.value() == deconv::kLimited
               && bwd_filter_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find a backward filter convolution algorithm " <<
              "(for use in deconvolution operator backprop-to-filter).";
        } else {
            this->back_algo_w_ = bwd_filter_algo[i].bwd_weights_algo;
        }

        miopenConvAlgoPerf_t bwd_data_algo[kMaxAlgos];
        /*CUDNN_CALL(miopenFindConvolutionBackwardDataAlgorithm(s->dnn_handle_,
                   filter_desc_,
                   in_desc_,
                   forward_conv_desc_,  // this backward algorithm used for inference
                   out_desc_,
                   kMaxAlgos,
                   &nalgo,
                   bwd_data_algo));*/ //TODO Temporarily commented.Parameters to be fixed
        i = 0;
        while (i < nalgo
               && (//bwd_data_algo[i].status != miopenStatusSuccess //TODO .Unsupported status variable in miopenConvAlgoPerf_t
                (param_.cudnn_tune.value() == deconv::kLimited
               && bwd_data_algo[i].memory > workspace_byte))) ++i;
        if (i == nalgo) {
          LOG(FATAL) << "Failed to find a backward data convolution algorithm." <<
              "(for use in deconvolution operator forward inference).";
        } else {
            this->back_algo_ = bwd_data_algo[i].bwd_data_algo;
        }
        CuDNNAlgoReg::Get()->Register(key, this->algo_, this->back_algo_,
                                      this->back_algo_w_);
      }
    }, ctx, {}, {var});
    Engine::Get()->WaitForVar(var);
    Engine::Get()->DeleteVariable([](RunContext s) {}, ctx, var);
  }

  void GetTempSize(const OpContext& ctx) {
    if (init_temp_size_) return;
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t back_size = 0, back_size_w = 0;
    CUDNN_CALL(miopenConvolutionBackwardDataGetWorkSpaceSize(s->dnn_handle_,
               in_desc_,
               filter_desc_,
               forward_conv_desc_,
               out_desc_,
               &back_size));
    CUDNN_CALL(miopenConvolutionBackwardWeightsGetWorkSpaceSize(s->dnn_handle_,
               in_desc_,
               out_desc_,
               backward_conv_desc_,
               filter_desc_,
               &back_size_w));
    backward_workspace_byte_ = std::max(back_size, back_size_w);
    CUDNN_CALL(miopenConvolutionForwardGetWorkSpaceSize(s->dnn_handle_,
               filter_desc_,
               out_desc_,
               backward_conv_desc_,
               in_desc_,
               &forward_workspace_byte_));

    forward_workspace_ = forward_workspace_byte_ / sizeof(DType) + 1;
    backward_workspace_ = backward_workspace_byte_ / sizeof(DType) + 1;
    init_temp_size_ = true;
  }

  bool init_cudnn_;
  bool init_temp_size_;
  size_t forward_workspace_;
  size_t backward_workspace_;
  size_t forward_workspace_byte_;
  size_t backward_workspace_byte_;
  size_t data_offset_;
  size_t out_offset_;
  size_t weight_offset_;
  size_t bias_offset_;
  miopenDataType_t dtype_;
  miopenTensorDescriptor_t in_desc_;
  miopenTensorDescriptor_t out_desc_;
  miopenTensorDescriptor_t bias_desc_;
  miopenTensorDescriptor_t filter_desc_;
  // Convolution descriptor for "forward" inference operation.
  // Note that in deconvolution, the forward operation is handled
  // by the cuDNN backprop-to-data kernel.
  miopenConvolutionDescriptor_t forward_conv_desc_;
  // Convolution descriptor for "back-prop" operations to data and filter.
  // Note that in deconvolution, the backprop-to-data operation is handled
  // by the cuDNN forward kernel, while the backprop-to-filter operation
  // stays consistent with the convolution operator and is handled by
  // the backprop-to-filter kernel.
  miopenConvolutionDescriptor_t backward_conv_desc_;
  // Algorithm for the cuDNN forward kernel (used in gradient backprop to input)
  miopenConvFwdAlgorithm_t algo_;
  // Algorithm for the cuDNN backprop-to-data kernel (used in inference)
  miopenConvBwdDataAlgorithm_t back_algo_;
  // Algorithm for the cuDNN backprop-to-filter kernel
  miopenConvBwdWeightsAlgorithm_t back_algo_w_;
  //cudnnTensorFormat_t format_;
  DeconvolutionParam param_;
};
#endif  // CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CUDNN_DECONVOLUTION_INL_H_
