#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/nlregularizedO_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void NLRegularizedOForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* W, const Dtype* Widx, const int channels,
    const int height, const int width, const int k, Dtype* top_data)
{
   CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    Dtype*  top_slice = top_data + (n * channels + c) * height * width;
	const Dtype*  bottom_slice = bottom_data + (n * channels + c) * height * width*k;

    int topidx = ph*width+pw;
	int bottomidx = topidx*k;// current idx in eta, eta is  (num,channel,height*width,k), W and Widx is (num,1,height*width,k)
	for(int i=0;i<k;i++)
	{
	  //bottomidx += i; 
	  top_slice[topidx] += W[bottomidx+i] * (bottom_slice[bottomidx+i]); // div_ui = sum Wi*(uij-uji)
	  for(int j=0;j<k;j++)
	  {
		if(int(Widx[int(Widx[bottomidx+i]*k+j)]) == topidx)
		{
			top_slice[topidx] -= W[bottomidx+i] * (bottom_slice[int(Widx[bottomidx+i]*k+j)]);
			break;
		}
		/*if(j==k-1)
		{
			//printf("n=%d, c=%d, ph = %d, pw= %d not found\n",n,c,ph,pw);
			printf("neighbour=%d, currentidx=%d, W[bottomidx+i]= %f, Widx[bottomidx+i] = %f, not found\n",bottomidx+i,topidx,Widx[int(Widx[bottomidx+i]*k+j)],Widx[bottomidx+i]);
		}*/
	  }
	}	
  }
}

template <typename Dtype>
void NLRegularizedOLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* eta = bottom[0]->gpu_data();
  const Dtype* W = bottom[1]->gpu_data();
  const Dtype* Widx = bottom[2]->gpu_data();
  const Dtype* O = bottom[3]->gpu_data();
  //const Dtype* lambda = this->blobs_[0]->cpu_data();
  //Dtype* mutable_lambda = this->blobs_[0]->mutable_cpu_data();

  Dtype* div_eta = div_eta_.mutable_gpu_data();

  Dtype* tilde_o = top[0]->mutable_gpu_data();

  int count = bottom[3]->count();

	//printf("W[0]=%f,Widx[0]=%f\n",*(bottom[1]->cpu_data()),*(bottom[2]->cpu_data()));

  // NOLINT_NEXT_LINE(whitespace/operators)
  NLRegularizedOForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, eta, W, Widx, channels_,
      height_, width_, k_,tilde_o);

  CUDA_CHECK(cudaMemcpy(div_eta, top[0]->gpu_data(), sizeof(Dtype) * count, cudaMemcpyDefault));

  //top0 = -divp
  caffe_gpu_scale(count, -Dtype(1), top[0]->gpu_data(), top[0]->mutable_gpu_data());
  //top0 = ok-divp
  caffe_gpu_axpy(count, Dtype(1), O, top[0]->mutable_gpu_data());
  //top0 = (ok-divp)/eps
  caffe_gpu_scale(count, Dtype(1/eps_), top[0]->gpu_data(), top[0]->mutable_gpu_data());
  iter_++;
  //if(iter_%100==1) std::cout<<"lambda: "<<lambda[0]<<" ";
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void NLRegularizedOBackward(const int nthreads, const Dtype* top_diff, const Dtype* W, const Dtype* Widx,
    const int channels, const int height, const int width, const int k, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //current pixel location
    const int pw = index % width;
    const int ph = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const Dtype*  top_slice = top_diff + (n * channels + c) * height * width;
    //direction = 0 represents x, direction = 1 represents y
	 Dtype*  bottom_slice = bottom_diff + (n * channels + c) * height * width*k;

    int topidx = ph*width+pw;
	int bottomidx = topidx*k;// current idx in xi, xi is  (num,channel,height*width,k), W and Widx is (num,1,height*width,k)

	for(int i=0;i<k;i++)// div_ui = sum Wi*(uij-uji)
	{
		bottom_slice[bottomidx+i] = W[bottomidx+i]*top_slice[topidx]; //Duij = Wi*Ddiv_dui - Wi*Ddiv_uj
	  for(int j=0;j<k;j++)
	  {
		if(int(Widx[int(Widx[bottomidx+i]*k+j)]) == topidx)
		{
			//bottom_slice[int(Widx[bottomidx+i]*k+j)] -= W[bottomidx+i]*top_slice[int(Widx[bottomidx+i])];
			bottom_slice[int(Widx[bottomidx+i]*k+j)] -= W[bottomidx+i]*top_slice[topidx];
			break;
		}
	  }
	}
  }
}


template <typename Dtype>
void NLRegularizedOLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const int count_eta = bottom[0]->count();
  const int count  = top[0]->count();
  const Dtype* bottom_W = bottom[1]->gpu_data();
  const Dtype* bottom_Widx = bottom[2]->gpu_data();

  Dtype* O_diff = bottom[3]->mutable_gpu_diff();
  Dtype* eta_diff = bottom[0]->mutable_gpu_diff();
  //std::cout<<"OK0"<<std::endl;
  //top0 = (ok-divp)/eps
  caffe_gpu_scale(count, Dtype(1/eps_), top_diff, O_diff);
  //caffe_copy(count,top_diff,O_diff);
  //CUDA_CHECK(cudaMemcpy(O_diff, top_diff, sizeof(Dtype)*count, cudaMemcpyDefault));
  //std::cout<<"OK1"<<std::endl;
  // NOLINT_NEXT_LINE(whitespace/operators)
  NLRegularizedOBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff,bottom_W,bottom_Widx, channels_,
      height_, width_, k_,eta_diff);
  //std::cout<<"OK2"<<std::endl;
  caffe_gpu_scale(count, -Dtype(1/eps_), bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  //std::cout<<"OK3"<<std::endl;
  /*
  Dtype* lambda_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* div_eta = div_eta_.gpu_data();
  Dtype result;
  caffe_gpu_dot(count, div_eta, top_diff, &result);
  *lambda_diff = -result;
  if(iter_%100==1) std::cout<<"lambda_diff: "<<lambda_diff[0]<<" ";
  */
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NLRegularizedOLayer);

}  // namespace caffe
