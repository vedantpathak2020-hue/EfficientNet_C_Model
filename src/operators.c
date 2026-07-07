#include "../include/operators.h"
#include <math.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

// Clamps a massive 32-bit integer safely back into an 8-bit limit
int8_t clamp_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// New clamp for 0 to 255 activation boundaries
uint8_t clamp_uint8(int32_t val) {
    if (val > 255) return 255;
    if (val < 0) return 0;
    return (uint8_t)val;
}

void apply_silu(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}

void apply_silu_int8(uint8_t *tensor, int size, float scale_in, int32_t zp_in, float scale_out, int32_t zp_out) {
    for (int i = 0; i < size; i++) {
        // Dequantize
        float x = ((float)tensor[i] - zp_in) * scale_in;
        
        // Apply SiLU
        float silu_val = x / (1.0f + expf(-x));
        
        // Requantize
        int32_t final_val = (int32_t)roundf(silu_val / scale_out) + zp_out;
        tensor[i] = clamp_uint8(final_val);
    }
}

void linear(const float *in, float *out, const float *weight, const float *bias, int in_c, int out_c) {
    for(int o = 0; o < out_c; o++) {
        float sum = bias ? bias[o] : 0.0f;
        for(int i = 0; i < in_c; i++) sum += in[i] * weight[o * in_c + i];
        out[o] = sum;
    }
}

// FIXED: Implements true PyTorch BatchNorm math
void batchnorm2d(float *data, const float *weight, const float *bias, 
                 const float *mean, const float *var, float eps, 
                 int channels, int h, int w) {
    int spatial_size = h * w;
    for (int c = 0; c < channels; c++) {
        float m = mean[c];
        float v = var[c];
        float w_val = weight[c];
        float b_val = bias[c];
        
        // Calculate the scale and shift for this channel once
        float inv_std = 1.0f / sqrtf(v + eps);
        float scale = w_val * inv_std;
        float shift = b_val - (m * scale);
        
        float *channel_ptr = data + (c * spatial_size);
        for (int i = 0; i < spatial_size; i++) {
            channel_ptr[i] = (channel_ptr[i] * scale) + shift;
        }
    }
}

void adaptive_avg_pool2d(float *in, float *out, int channels, int h, int w) {
    int spatial_size = h * w;
    float inv_spatial = 1.0f / (float)spatial_size;
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; i++) sum += in[c * spatial_size + i];
        out[c] = sum * inv_spatial;
    }
}

void conv2d(float *in, float *out, const float *weight, const float *bias, 
            int in_c, int out_c, int h, int w, int k, int stride) {
    int pad = k / 2;
    int out_h = (h + 2 * pad - k) / stride + 1;
    int out_w = (w + 2 * pad - k) / stride + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ki = 0; ki < k; ki++) {
                        for (int kj = 0; kj < k; kj++) {
                            int in_i = i * stride + ki - pad;
                            int in_j = j * stride + kj - pad;
                            if (in_i >= 0 && in_i < h && in_j >= 0 && in_j < w) {
                                sum += in[(ic * h + in_i) * w + in_j] * weight[((oc * in_c + ic) * k + ki) * k + kj];
                            }
                        }
                    }
                }
                out[(oc * out_h + i) * out_w + j] = sum;
            }
        }
    }
}

void conv2d_int8(const uint8_t *in, uint8_t *out, const int8_t *wt, const float *bias, 
                 int in_c, int out_c, int h, int w, int k, int stride,
                 float scale_in, int32_t zp_in,
                 float scale_wt, int32_t zp_wt,
                 float scale_out, int32_t zp_out) 
{
    int pad = k / 2;
    int out_h = (h + 2 * pad - k) / stride + 1;
    int out_w = (w + 2 * pad - k) / stride + 1;

    // The Multiplier: This single float perfectly converts the 32-bit sum back to 8-bit
    float M = (scale_in * scale_wt) / scale_out;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                
                int32_t accumulator = 0; // The 32-bit safe zone
                
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;

                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                int in_idx = (ic * h * w) + (ih * w) + iw;
                                int wt_idx = (oc * in_c * k * k) + (ic * k * k) + (kh * k) + kw;
                                
                                // Mathematically strip the zero-points BEFORE multiplying
                                int32_t in_val = in[in_idx] - zp_in;
                                int32_t wt_val = wt[wt_idx] - zp_wt;
                                
                                accumulator += in_val * wt_val;
                            }
                        }
                    }
                }
                
                // --- THE RE-QUANTIZATION PHASE ---
                // 1. Scale the massive integer down
                float scaled_val = (float)accumulator * M; 
                
                // 2. Add the bias (PyTorch usually leaves biases as unquantized floats)
                if (bias != NULL) {
                   scaled_val += (bias[oc] / scale_out); 
                }
                
                // 3. Shift by the Output Zero-Point and clamp it to INT8
                int32_t final_val = (int32_t)roundf(scaled_val) + zp_out;
                
                int out_idx = (oc * out_h * out_w) + (oh * out_w) + ow;
                out[out_idx] = clamp_uint8(final_val);
            }
        }
    }
}

void conv2d_depthwise(float *in, float *out, const float *weight, 
                      int channels, int h, int w, int k, int stride) {
    int pad = k / 2;
    int out_h = (h + 2 * pad - k) / stride + 1;
    int out_w = (w + 2 * pad - k) / stride + 1;

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < k; ki++) {
                    for (int kj = 0; kj < k; kj++) {
                        int in_i = i * stride + ki - pad;
                        int in_j = j * stride + kj - pad;
                        if (in_i >= 0 && in_i < h && in_j >= 0 && in_j < w) {
                            sum += in[(c * h + in_i) * w + in_j] * weight[(c * k + ki) * k + kj];
                        }
                    }
                }
                out[(c * out_h + i) * out_w + j] = sum;
            }
        }
    }
}

void conv2d_depthwise_int8(const uint8_t *in, uint8_t *out, const int8_t *wt, const float *bias, 
                           int channels, int h, int w, int k, int stride,
                           float scale_in, int32_t zp_in,
                           float scale_wt, int32_t zp_wt,
                           float scale_out, int32_t zp_out) 
{
    int pad = k / 2;
    int out_h = (h + 2 * pad - k) / stride + 1;
    int out_w = (w + 2 * pad - k) / stride + 1;

    // The universal requantization multiplier
    float M = (scale_in * scale_wt) / scale_out;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                
                int32_t accumulator = 0; // 32-bit safe zone
                
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;

                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                            int in_idx = (c * h * w) + (ih * w) + iw;
                            int wt_idx = (c * k * k) + (kh * k) + kw;
                            
                            int32_t in_val = in[in_idx] - zp_in;
                            int32_t wt_val = wt[wt_idx] - zp_wt;
                            
                            accumulator += in_val * wt_val;
                        }
                    }
                }
                
                // Re-quantize back to 8-bit
                float scaled_val = (float)accumulator * M; 
                if (bias != NULL) {
                   scaled_val += (bias[c] / scale_out); 
                }
                
                int32_t final_val = (int32_t)roundf(scaled_val) + zp_out;
                
                int out_idx = (c * out_h * out_w) + (oh * out_w) + ow;
                out[out_idx] = clamp_uint8(final_val);
            }
        }
    }
}

void skip_add_int8(uint8_t *tensor_a, const uint8_t *tensor_b, int size,
                   float scale_a, int32_t zp_a,
                   float scale_b, int32_t zp_b,
                   float scale_out, int32_t zp_out) 
{
    for (int i = 0; i < size; i++) {
        // 1. Dequantize both incoming streams
        float val_a = ((float)tensor_a[i] - zp_a) * scale_a;
        float val_b = ((float)tensor_b[i] - zp_b) * scale_b;
        
        // 2. Perform the true addition
        float sum = val_a + val_b;
        
        // 3. Re-quantize to the new output scale
        int32_t final_val = (int32_t)roundf(sum / scale_out) + zp_out;
        tensor_a[i] = clamp_uint8(final_val);
    }
}

void squeeze_excitation(float *data, int channels, int h, int w, int reduced_c, 
                        const float *w1, const float *b1, const float *w2, const float *b2) {
    float pool[1280] = {0}, reduced[1280] = {0}, expanded[1280] = {0};
    
    adaptive_avg_pool2d(data, pool, channels, h, w);
    linear(pool, reduced, w1, b1, channels, reduced_c);
    apply_silu(reduced, reduced_c);
    linear(reduced, expanded, w2, b2, reduced_c, channels);
    
    int spatial = h * w;
    for(int c = 0; c < channels; c++) {
        float scale = 1.0f / (1.0f + expf(-expanded[c])); 
        for(int i = 0; i < spatial; i++) {
            data[c * spatial + i] *= scale;
        }
    }
}

// Static buffers for the SE block (Max channels in EfficientNet-B0 is ~1920)
static float se_pool[2000];
static float se_reduced[2000];
static float se_expand[2000];

void squeeze_excitation_int8(uint8_t *tensor, int channels, int h, int w, int reduced_c,
                             const int8_t *w1, const float *b1, float s_w1, int32_t z_w1,
                             const int8_t *w2, const float *b2, float s_w2, int32_t z_w2,
                             float s_in, int32_t z_in, 
                             float s_out, int32_t z_out)
{
    // 1. SQUEEZE: Global Average Pooling (Calculate in float for precision)
    for (int c = 0; c < channels; c++) {
        int32_t sum = 0;
        for (int i = 0; i < h * w; i++) {
            sum += (tensor[(c * h * w) + i] - z_in);
        }
        float avg = (float)sum / (h * w);
        se_pool[c] = avg * s_in; // Convert back to true float value
    }

    // 2. EXCITATION LAYER 1: Reduce & SiLU
    for (int rc = 0; rc < reduced_c; rc++) {
        float acc = 0;
        for (int c = 0; c < channels; c++) {
            float weight_val = (w1[rc * channels + c] - z_w1) * s_w1;
            acc += se_pool[c] * weight_val;
        }
        if (b1 != NULL) acc += b1[rc];
        
        // SiLU Activation
        se_reduced[rc] = acc / (1.0f + expf(-acc)); 
    }

    // 3. EXCITATION LAYER 2: Expand & Sigmoid
    for (int c = 0; c < channels; c++) {
        float acc = 0;
        for (int rc = 0; rc < reduced_c; rc++) {
            float weight_val = (w2[c * reduced_c + rc] - z_w2) * s_w2;
            acc += se_reduced[rc] * weight_val;
        }
        if (b2 != NULL) acc += b2[c];
        
        // Sigmoid Activation
        se_expand[c] = 1.0f / (1.0f + expf(-acc)); 
    }

    // 4. THE MULTIPLIER: Apply the attention weights and Requantize
    for (int c = 0; c < channels; c++) {
        float attention = se_expand[c];
        for (int i = 0; i < h * w; i++) {
            int idx = (c * h * w) + i;
            
            // Dequantize -> Multiply by Attention -> Requantize
            float original_val = (tensor[idx] - z_in) * s_in;
            float scaled_val = original_val * attention;
            
            int32_t final_val = (int32_t)roundf(scaled_val / s_out) + z_out;
            tensor[idx] = clamp_uint8(final_val);
        }
    }
}

float cross_entropy_loss(float *logits, int target_class, int num_classes) {
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    float log_prob = (logits[target_class] - max_logit) - logf(sum_exp);
    return -log_prob;
}

// The Entry Door: Converts standard float pixels into INT8 mapped values
void quantize_tensor(const float *in, uint8_t *out, int size, float scale, int32_t zp) {
    for (int i = 0; i < size; i++) {
        int32_t val = (int32_t)roundf(in[i] / scale) + zp;
        out[i] = clamp_uint8(val);
    }
}

// The Exit Door: Converts the final INT8 network outputs back to usable Floats
void dequantize_tensor(const uint8_t *in, float *out, int size, float scale, int32_t zp) {
    for (int i = 0; i < size; i++) {
        out[i] = ((float)in[i] - (float)zp) * scale;
    }
}