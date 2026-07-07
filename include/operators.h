#ifndef OPERATORS_H
#define OPERATORS_H

#include <stdint.h>

void apply_silu(float *data, int size);

void apply_silu_int8(uint8_t *tensor, int size, float scale_in, int32_t zp_in, float scale_out, int32_t zp_out);

void conv2d(float *in, float *out, const float *weight, const float *bias, 
            int in_c, int out_c, int h, int w, int k, int stride);

void conv2d_int8(const uint8_t *in, uint8_t *out, const int8_t *wt, const float *bias, 
                 int in_c, int out_c, int h, int w, int k, int stride,
                 float scale_in, int32_t zp_in,
                 float scale_wt, int32_t zp_wt,
                 float scale_out, int32_t zp_out);
                            
void conv2d_depthwise(float *in, float *out, const float *weight, 
                      int channels, int h, int w, int k, int stride);

void conv2d_depthwise_int8(const uint8_t *in, uint8_t *out, const int8_t *wt, const float *bias, 
                           int channels, int h, int w, int k, int stride,
                           float scale_in, int32_t zp_in,
                           float scale_wt, int32_t zp_wt,
                           float scale_out, int32_t zp_out);

// FIXED: Now includes running_mean, running_var, and eps!
void batchnorm2d(float *data, const float *weight, const float *bias, 
                 const float *mean, const float *var, float eps, 
                 int channels, int h, int w);

void adaptive_avg_pool2d(float *in, float *out, int channels, int h, int w);

void linear(const float *in, float *out, const float *weight, const float *bias, 
            int in_c, int out_c);

void squeeze_excitation(float *data, int channels, int h, int w, int reduced_c, 
                        const float *w1, const float *b1, const float *w2, const float *b2);

void squeeze_excitation_int8(uint8_t *tensor, int channels, int h, int w, int reduced_c,
                             const int8_t *w1, const float *b1, float s_w1, int32_t z_w1,
                             const int8_t *w2, const float *b2, float s_w2, int32_t z_w2,
                             float s_in, int32_t z_in, 
                             float s_out, int32_t z_out);

// Calculates Cross-Entropy Loss for a single prediction
float cross_entropy_loss(float *logits, int target_class, int num_classes);

void quantize_tensor(const float *in, uint8_t *out, int size, float scale, int32_t zp);

void dequantize_tensor(const uint8_t *in, float *out, int size, float scale, int32_t zp);

void skip_add_int8(uint8_t *tensor_a, const uint8_t *tensor_b, int size,
                   float scale_a, int32_t zp_a,
                   float scale_b, int32_t zp_b,
                   float scale_out, int32_t zp_out);

                   
#endif // OPERATORS_H