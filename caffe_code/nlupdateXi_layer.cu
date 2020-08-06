#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/nlupdateXi_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void NLUpdateXiForward(const int nthreads,
    const Dtype*  bottom_data, const int channels,
    const int height, const int width, const int k, const Dtype*  W, const Dtype*  Widx, Dtype*  top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype*  bottom_slice = bottom_data + (n * channels + c) * height * width;
    //direction = 0 represents x, direction = 1 represents y
    Dtype*  top_slice = top_data + (n * channels + c) * height * width*k;

    int bottomidx = ph*width+pw;
	int topidx = bottomidx*k;// current idx in xi, xi is  (num,channel,height*width,k), W and Widx is (num,1,height*width,k)
	for(int i=0;i<k;i++)
	{
	  //topidx += i; 
	  top_slice[topidx+i] = W[topidx+i] * ( bottom_slice[int(Widx[topidx+i])] - bottom_slice[bottomidx] ); // gradient_uij = w*(uj-ui)
	}	
  }
}

template <typename Dtype>
void NLUpdateXiLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // xi(t) = xi(t-1) - tau*lambda*grad(A(t-1))
  const Dtype* bottom_data = bottom[0]->gpu_data(); //A(t-1)
  const Dtype* bottom_W;//weight matrix W
  const Dtype* bottom_Widx;
  const Dtype* bottom_xi;// Xi

  bottom_W = bottom[1]->gpu_data();
  bottom_Widx = bottom[2]->gpu_data();
  if(bottom.size()==4)
  {
    bottom_xi = bottom[3]->gpu_data();
  }
  //const Dtype* tau = this->blobs_[0]->cpu_data();
  //Dtype* mutable_tau = this->blobs_[0]->mutable_cpu_data();
  
  Dtype* top_xi = top[0]->mutable_gpu_data();

  int count = bottom[0]->count();
  int tcount = top[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  NLUpdateXiForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_,
      height_, width_, k_,bottom_W, bottom_Widx,top_xi);
  caffe_gpu_scale(tcount, Dtype(-tau_), top[0]->gpu_data(), top[0]->mutable_gpu_data());

  if(bottom.size()==4)
  {
    caffe_gpu_add(tcount, bottom_xi, top[0]->gpu_data(), top[0]->mutable_gpu_data());
  }

  //iter_++;
  //if(iter_%100==1) std::cout<<"tau: "<<tau[0]<<" ";
  //caffe_gpu_scale(count, -lambda[0]*tau, top[0]->gpu_data(), top[0]->mutable_gpu_data());
  //caffe_gpu_scale(count, -lambda[0]*tau, top[1]->gpu_data(), top[1]->mutable_gpu_data());
  
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void NLUpdateXiBackward(const int nthreads, const int channels, const int height, const int width, const int k, const Dtype*  W, const Dtype*  Widx, const Dtype*  top_diff, Dtype*  bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    Dtype*  bottom_slice = bottom_diff + (n * channels + c) * height * width;
    //direction = 0 represents x, direction = 1 represents y
	const Dtype*  top_slice = top_diff + (n * channels + c) * height * width*k;

    int bottomidx = ph*width+pw;
	int topidx = bottomidx*k;// current idx in xi, xi is  (num,channel,height*width,k), W and Widx is (num,1,height*width,k)

	for(int i=0;i<k;i++)// gradient_uij = w*(uj-ui)
	{
		bottom_slice[bottomidx] -= W[topidx+i]*top_slice[topidx+i];
		bottom_slice[int(Widx[topidx+i])] += W[topidx+i]*top_slice[topidx+i];
	}
  }
}


template <typename Dtype>
void NLUpdateXiLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int count_xi  = top[0]->count();
  const Dtype* bottom_W = bottom[1]->gpu_data();
  const Dtype* bottom_Widx = bottom[2]->gpu_data();
  //const Dtype* lambda = bottom[1]->cpu_data();
  //const Dtype* tau = this->blobs_[0]->cpu_data();

  //Blob<Dtype>* temp_bp0, *temp_bp1;
  //temp_bp0->ReshapeLike(*bottom[0]);
  //temp_bp1->ReshapeLike(*bottom[0]);
  //std::cout<<"temp_bp0 count: "<<temp_bp0.count()<<std::endl;
  //cudaMemcpy(temp_bp0.mutable_gpu_diff(), top[0]->gpu_diff(), sizeof(Dtype)*count, cudaMemcpyDefault);

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  //std::cout<<"bottom[0] count: "<<bottom[0]->count()<<std::endl;
  // NOLINT_NEXT_LINE(whitespace/operators)
  NLUpdateXiBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels_, height_, width_, k_,bottom_W, bottom_Widx,top_diff, bottom_diff);
  caffe_gpu_scale(count, Dtype(-tau_), bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());

  if(bottom.size()==4)
  {
	//std::cout<<"bottom size: "<<bottom.size()<<std::endl;
    cudaMemcpy(bottom[3]->mutable_gpu_diff(),top_diff, sizeof(Dtype)*count_xi, cudaMemcpyDefault);
  }

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NLUpdateXiLayer);

}  // namespace caffe
