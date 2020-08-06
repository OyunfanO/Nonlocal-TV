#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/nlregularizedO_layer.hpp"

namespace caffe {

template <typename Dtype>
void NLRegularizedOLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //initialize tau		  
  //this->blobs_.resize(1);
  //vector<int> lambda_shape(1,1);
  //this->blobs_[0].reset(new Blob<Dtype>(lambda_shape));
  //shared_ptr<Filler<Dtype> > lambda_filler(GetFiller<Dtype>(this->layer_param_.divergence_param().lambda_filler()));
  //lambda_filler->Fill(this->blobs_[0].get());
  iter_ = 0;
  NLRegularizedoParameter nlregularizedo_param = this->layer_param_.nlregularizedo_param();
  eps_ = nlregularizedo_param.eps();//number of TV_bias filters
}

template <typename Dtype>
void NLRegularizedOLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//bottom[0] is eta, bottom[1] is W, bottom[2] is Widx, bottom[3] is O
  channels_ = bottom[0]->channels();
  num_ = bottom[0]->num();
  k_ = bottom[0]->width();
  height_ = bottom[3]->height();
  width_ = bottom[3]->width();

  top[0]->ReshapeLike(*bottom[3]);
  
  vector<int> div_eta_shape(1, bottom[3]->count());
  div_eta_.Reshape(div_eta_shape);
  caffe_set(bottom[3]->count(), Dtype(0),div_eta_.mutable_cpu_data());
  caffe_set(bottom[3]->count(), Dtype(0),top[0]->mutable_cpu_data());
}

template <typename Dtype>
void NLRegularizedOLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void NLRegularizedOLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(NLRegularizedOLayer);
#endif

INSTANTIATE_CLASS(NLRegularizedOLayer);
REGISTER_LAYER_CLASS(NLRegularizedO);
}  // namespace caffe
