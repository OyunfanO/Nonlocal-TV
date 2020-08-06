#include <algorithm>
#include <cfloat>
#include <vector>
#include <glog/logging.h>

#include "caffe/layer.hpp"
#include "caffe/layers/NLTV_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {  

template <typename Dtype>
void NLTVLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  nlupdatexi_layer_->SetUp(nlupdatexi_bottom_vec_, nlupdatexi_top_vec_);
  nlprojection_layer_->SetUp(nlprojection_bottom_vec_, nlprojection_top_vec_);
  nlregularizedo_layer_->SetUp(nlregularizedo_bottom_vec_, nlregularizedo_top_vec_);

}

template <typename Dtype>
void NLTVLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    const float* label_count_data = 
        weight_by_label_freqs_ ? label_counts_.gpu_data() : NULL;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label,
        weight_by_label_freqs_, label_count_data, bottom_diff, 
        outer_num_, dim, inner_num_, has_ignore_label_, 
        ignore_label_, counts);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NLTVLayer);

}  // namespace caffe
