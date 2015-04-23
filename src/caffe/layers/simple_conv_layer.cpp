#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SimpleConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void SimpleConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight_data = this->blobs_[0]->cpu_data();
  const Dtype* bias_data = this->blobs_[1]->cpu_data();

  //
  // Apply the weights
  //
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    for (int n = 0; n < this->num_; n++) {
      int o_g = this->channels_ / this->group_;
      int k_g = bottom[i]->channels_ / this->group_;
      for (int g = 0; g < groups; g++) {
        int o_head = o_g * g;
        int k_head = k_g * g;
        for (int o = 0; o < o_g; o++) {
          for (int k = 0; k < k_g; k++) {
            for (int y = 0; y < top[i]->height(); y++) {
              for (int x = 0; x < top[i]->width(); x++) {
                for (int p = 0; p < kernel_h; p++) {
                  for (int q = 0; q < kernel_w; q++) {
                    int in_y = y * this->stride_h_ - this->pad_h_ + p;
                    int in_x = x * this->stride_w_ - this->pad_w_ + q;
                    if (in_y >= 0 && in_y < this->height_
                      && in_x >= 0 && in_x < this->width_) {
                      top_data[top[i]->offset(n, o + o_head, y, x)] +=
                          bottom_data[bottom[i]->offset(n, k + k_head, in_y, in_x)]
                          * weight_data[this->blobs_[0]->offset(o + o_head, k, p, q)];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  
    //
    // Add the bias
    //
    if (this->bias_term()) {
      for (int n = 0; n < top[i]->num(); n++) {
        for (int o = 0; o < top[i]->channels_; o++) {
          for (int y = 0; y < top[i]->height(); y++) {
            for (int x = 0; x < top[i]->width(); x++) {
              top_data[top[i]->offset(n, o, y, x)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void SimpleConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SimpleConvolutionLayer);
#endif

INSTANTIATE_CLASS(SimpleConvolutionLayer);

}  // namespace caffe
