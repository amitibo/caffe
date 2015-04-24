#include <vector>
#include <iostream>

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
void SimpleConvolutionLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top
  ) {

  const Dtype* weight_data = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    //
    // Clear the top
    //
    caffe_set(top[i]->count(), Dtype(0), top_data);

    DLOG(INFO) << "Applying weights to data: " << i;
    for (int n = 0; n < this->num_; n++) {
      int o_g = this->num_output_ / this->group_;
      int k_g = this->channels_ / this->group_;
      for (int g = 0; g < this->group_; g++) {
        int o_head = o_g * g;
        int k_head = k_g * g;
        for (int o = 0; o < o_g; o++) {
          for (int k = 0; k < k_g; k++) {
            for (int y = 0; y < this->height_out_; y++) {
              for (int x = 0; x < this->width_out_; x++) {
                for (int p = 0; p < this->kernel_h_; p++) {
                  for (int q = 0; q < this->kernel_w_; q++) {
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
    if (this->bias_term_) {
      const Dtype* bias_data = this->blobs_[1]->cpu_data();
      
      DLOG(INFO) << "Applying bias to data: " << i;
      for (int n = 0; n < this->num_; n++) {
        for (int o = 0; o < this->num_output_; o++) {
          for (int y = 0; y < this->height_out_; y++) {
            for (int x = 0; x < this->width_out_; x++) {
              top_data[top[i]->offset(n, o, y, x)] +=
                bias_data[o];
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void SimpleConvolutionLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom
  ) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  
  //
  // Clear the weight diff.
  //
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  
  //
  // Clear the bias diff.
  //
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    
    //
    // Bias gradient, if necessary.
    //
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      
      DLOG(INFO) << "Applying gradient bias to data: " << i;
      for (int n = 0; n < this->num_; n++) {
        for (int o = 0; o < this->num_output_; o++) {
          for (int y = 0; y < this->height_out_; y++) {
            for (int x = 0; x < this->width_out_; x++) {
              bias_diff[o] +=
                top_diff[top[i]->offset(n, o, y, x)];
            }
          }
        }
      }
    }
    
    if (this->param_propagate_down_[0] || propagate_down[i]) {

      //
      // gradient w.r.t. bottom data, if necessary.
      //
      for (int n = 0; n < this->num_; n++) {
        int o_g = this->num_output_ / this->group_;
        int k_g = this->channels_ / this->group_;
        for (int g = 0; g < this->group_; g++) {
          int o_head = o_g * g;
          int k_head = k_g * g;
          for (int o = 0; o < o_g; o++) {
            for (int k = 0; k < k_g; k++) {
              for (int y = -this->pad_h_; y < this->height_out_ + this->pad_h_; y++) {
                for (int x = -this->pad_w_; x < this->width_out_ + this->pad_w_; x++) {
                  for (int p = this->kernel_h_; p --> 0 ;) {
                    for (int q = this->kernel_w_; q --> 0 ;) {
                      int in_y = y * this->stride_h_ - this->pad_h_ + this->kernel_h_ - p + 1;
                      int in_x = x * this->stride_w_ - this->pad_w_ + this->kernel_w_ - q + 1;
                      if (y >= 0 && y < this->height_out_
                        && x >= 0 && x < this->width_out_) {
                        bottom_diff[bottom[i]->offset(n, k + k_head, in_y, in_x)] +=
                            top_diff[top[i]->offset(n, o +o_head, y, x)]
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
