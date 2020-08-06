#include <algorithm>
#include <vector>

#include "caffe/layers/nlprojection_layer.hpp"
#include <cmath>
#include <iostream>
#include <stdio.h>

namespace caffe {

template <typename Dtype>
__global__ void RowASum(const int n, const float lambda, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
	//idx = index%(channels_*height_*k_);
	out[index] = in[index] > lambda ? lambda/in[index] : 1;
	//printf("idxadd=%x,index=%d, out[index]=%f \n",out+index,index,out[index]);
  }
}


template <typename Dtype>
__global__ void NLProjectionForward(const int n, const int k, const Dtype* xi, const Dtype* nrow, Dtype* eta) {
  CUDA_KERNEL_LOOP(index, n) {
	for(int i=0;i<k;i++)
	{
		eta[index*k+i] = xi[index*k+i]*nrow[index];
	}
  }
}

template <typename Dtype>
void NLProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* xi = bottom[0]->gpu_data();

  Dtype* nxi_mu = norm_xi->mutable_gpu_data();
  const Dtype* nxi = norm_xi->gpu_data();

  Dtype* nrow_mu = norm_row->mutable_gpu_data();
  const Dtype* nrow = norm_row->gpu_data();

  Dtype* eta_mu = top[0]->mutable_gpu_data();

  const Dtype* multi = multiplier->gpu_data();

  const int count = bottom[0]->count();

  caffe_gpu_powx(count,xi,Dtype(2),nxi_mu);
  caffe_gpu_gemv(CblasNoTrans, num_*channels_*height_, k_, Dtype(1), nxi, multi, Dtype(0), nrow_mu);
  caffe_gpu_powx( num_*channels_*height_,nrow,Dtype(0.5),nrow_mu);

  RowASum<Dtype><<<CAFFE_GET_BLOCKS(num_*channels_*height_), CAFFE_CUDA_NUM_THREADS>>>(
      num_*channels_*height_, lambda_, norm_row->gpu_data(), norm_row->mutable_gpu_data());

  NLProjectionForward<Dtype><<<CAFFE_GET_BLOCKS(num_*channels_*height_), CAFFE_CUDA_NUM_THREADS>>>(
	  num_*channels_*height_, k_, xi, nrow, eta_mu);


  /* 
  //x1_s = xi1^2, x2_s = xi2^2
  caffe_gpu_mul(count,xi1,xi1,xi1_s->mutable_gpu_data());
  caffe_gpu_mul(count,xi2,xi2,xi2_s->mutable_gpu_data());
  //norm_xi = xi1^2+xi2^2
  caffe_gpu_add(count, xi1_s->cpu_data(),xi2_s->cpu_data(),norm_xi->mutable_cpu_data());
  //norm_xi = sqrt(xi1^2+xi2^2)
  caffe_gpu_powx(count,norm_xi->gpu_data(),Dtype(0.5),norm_xi->mutable_gpu_data());

  ProjectionForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, xi1, xi2, eta1, eta2, norm_xi->gpu_data());
  */

  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void NLProjectionBackward(const int n, const int k, Dtype* bottom_diff, const Dtype* nrow,
const Dtype* top_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    //caffe_gpu_scale(k,Dtype(1/sum),top_diff+index*k,bottom_diff+index*k);
	for(int i=0;i<k;i++)
	{
		bottom_diff[index*k+i] = top_diff[index*k+i]*nrow[index];
	}
  }
}

template <typename Dtype>
void NLProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* eta_diff = top[0]->gpu_diff();
  const Dtype* nrow = norm_row->gpu_data();
  Dtype* xi_diff = bottom[0]->mutable_gpu_diff();

  NLProjectionBackward<Dtype><<<CAFFE_GET_BLOCKS(num_*channels_*height_), CAFFE_CUDA_NUM_THREADS>>>(
	  num_*channels_*height_, k_, xi_diff, nrow, eta_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(NLProjectionLayer);


}  // namespace caffe
