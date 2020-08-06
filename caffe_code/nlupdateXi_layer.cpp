#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/nlupdateXi_layer.hpp"

namespace caffe {

template <typename Dtype>
void NLUpdateXiLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //initialize tau
  /*		  
  this->blobs_.resize(1);
  vector<int> tau_shape(1,1);
  this->blobs_[0].reset(new Blob<Dtype>(tau_shape));
  shared_ptr<Filler<Dtype> > tau_filler(GetFiller<Dtype>(this->layer_param_.gradient_param().tau_filler()));
  tau_filler->Fill(this->blobs_[0].get());
  iter_ = 0;
  */
  NLUpdatexiParameter nlupdatexi_param = this->layer_param_.nlupdatexi_param();
  tau_ = nlupdatexi_param.tau();//step size
  //lambda_ = nlupdatexi_param.lambda();
}


template <typename Dtype>
void NLUpdateXiLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //bottom[0] is A, bottom[1] is W, bottom[2] is Widx, bottom[3] is xi
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();
  k_ = bottom[1]->width();

  vector<int> gradient_shape(4);
  gradient_shape[0] = num_;
  gradient_shape[1] = channels_;
  gradient_shape[2] = height_*width_;
  gradient_shape[3] = k_;

  top[0]->Reshape(gradient_shape);  
  caffe_set(top[0]->count(), Dtype(0),top[0]->mutable_cpu_data());

/*
  temp_bp0.ReshapeLike(*bottom[0]);
  temp_bp1.ReshapeLike(*bottom[0]);
  caffe_set(temp_bp0.count(), Dtype(0),temp_bp0.mutable_cpu_data());
  caffe_set(temp_bp1.count(), Dtype(0),temp_bp1.mutable_cpu_data());
  for(int top_id=0;top_id<top.size();top_id++)
  {
    top[top_id]->ReshapeLike(*bottom[0]);
  }*/
  
  /*
  vector<int> tau_multiplier_shape(1, bottom[0]->count());
  tau_multiplier_.Reshape(tau_multiplier_shape);
  caffe_set(tau_multiplier_.count(), Dtype(1),tau_multiplier_.mutable_cpu_data());
  */
}

template <typename Dtype>
void NLUpdateXiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void NLUpdateXiLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}


#ifdef CPU_ONLY
STUB_GPU(NLUpdateXiLayer);
#endif

INSTANTIATE_CLASS(NLUpdateXiLayer);
REGISTER_LAYER_CLASS(NLUpdateXi);
}  // namespace caffe
