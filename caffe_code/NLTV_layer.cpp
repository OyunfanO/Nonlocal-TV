#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/NLTV_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NLTVLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//bottom[0] ro1, bottom[1] xi1, bottom[2] W, bottom[3] Widx, bottom[4] O
  Layer<Dtype>::LayerSetUp(bottom, top);

  NLTV nltv_param = this->layer_param_.nltv_param();
  iter_ = nltv_param.iter();

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&A);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  LayerParameter nlupdatexi_param(this->layer_param_);
  nlupdatexi_param.set_type("NLUpdatexi");
  nlupdatexi_layer_ = LayerRegistry<Dtype>::CreateLayer(nlupdatexi_param);
  nlupdatexi_bottom_vec_.clear();
  nlupdatexi_bottom_vec_.push_back(&A);
  nlupdatexi_bottom_vec_.push_back(bottom[2]);
  nlupdatexi_bottom_vec_.push_back(bottom[3]);
  nlupdatexi_bottom_vec_.push_back(bottom[1]);
  nlupdatexi_top_vec_.clear();
  nlupdatexi_top_vec_.push_back(&Xi);
  nlupdatexi_layer_->SetUp(nlupdatexi_bottom_vec_, nlupdatexi_top_vec_);

  LayerParameter nlprojection_param(this->layer_param_);
  nlprojection_param.set_type("NLProjection");
  nlprojection_layer_ = LayerRegistry<Dtype>::CreateLayer(nlprojection_param);
  nlprojection_bottom_vec_.clear();
  nlprojection_bottom_vec_.push_back(&Xi);
  nlprojection_top_vec_.clear();
  nlprojection_top_vec_.push_back(&Eta);
  nlprojection_layer_->SetUp(nlprojection_bottom_vec_, nlprojection_top_vec_);

  LayerParameter nlregularizedo_param(this->layer_param_);
  nlregularizedo_param.set_type("NLRegularizedo");
  nlregularizedo_layer_ = LayerRegistry<Dtype>::CreateLayer(nlregularizedo_param);
  nlregularizedo_bottom_vec_.clear();
  nlregularizedo_bottom_vec_.push_back(&Eta);
  nlregularizedo_bottom_vec_.push_back(bottom[2]);
  nlregularizedo_bottom_vec_.push_back(bottom[3]);
  nlregularizedo_bottom_vec_.push_back(bottom[4]);
  nlregularizedo_top_vec_.clear();
  nlregularizedo_top_vec_.push_back(&Ro);
  nlregularizedo_layer_->SetUp(nlregularizedo_bottom_vec_, nlregularizedo_top_vec_);


}

template <typename Dtype>
void NLTVLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  nlupdatexi_layer_->Reshape(nlupdatexi_bottom_vec_, nlupdatexi_top_vec_);
  nlprojection_layer_->Reshape(nlprojection_bottom_vec_, nlprojection_top_vec_);
  nlregularizedo_layer_->Reshape(nlregularizedo_bottom_vec_, nlregularizedo_top_vec_);

  channels_ = bottom[0]->channels();
  num_ = bottom[0]->num();
  k_ = bottom[2]->width();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NLTVLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /*
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      const int idx = i * dim + label_value * inner_num_ + j;
      if (weight_by_label_freqs_) {
        const float* label_count_data = label_counts_.cpu_data();
        loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN)))
            * static_cast<Dtype>(label_count_data[label_value]);
      } else {
        loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN)));
      }
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }*/
}

template <typename Dtype>
void NLTVLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /*
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    const float* label_count_data = 
        weight_by_label_freqs_ ? label_counts_.cpu_data() : NULL;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          const int idx = i * dim + label_value * inner_num_ + j;
          bottom_diff[idx] -= 1;
          if (weight_by_label_freqs_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *= static_cast<Dtype>(label_count_data[label_value]);
            }
          }
          ++count;
        }
      }
    }
    // Scale gradient loss_weight should be always 1
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(NLTVLayer);
#endif

INSTANTIATE_CLASS(NLTVLayer);
REGISTER_LAYER_CLASS(NLTVLoss);

}  // namespace caffe
