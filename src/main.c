#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "../include/operators.h"

#define MAX_FEATURES 131072
#define NUM_TEST_IMAGES 10000
#define NUM_CLASSES 10
#define MAX_ROM_SIZE 5500000 // 5.5 MB static ROM block

typedef struct {
    float scale;
    int32_t zero_point;
    const int8_t *weights;
    float out_scale;
    int32_t out_zero_point;
    const float *bias; // <-- THE SAVIOR
} QTensor;

// --- HIGH-PRECISION POSIX TIMERS ---
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// --- GLOBAL OP-TYPE PROFILING TRACKERS ---
double acc_pw_ms = 0.0;
double acc_dw_ms = 0.0;
double acc_se_ms = 0.0;
double acc_pool_fc_ms = 0.0;

// --- LAYER-BY-LAYER TRACKING STRUCTS ---
typedef struct {
    char name[32];
    double time_ms;
    long alloc_bytes;
} LayerProfile;

LayerProfile profiles[50];
int num_profiles = 0;
long global_heap_diff = 0;

// --- PROFILING MACROS (Updated for Silent Tracking) ---
#define PROFILE_START() do { if (img_idx == 0) layer_start = get_time_ms(); } while(0)

#define PROFILE_END(layer_name, out_c, out_h, out_w) do { \
    if (img_idx == 0) { \
        layer_end = get_time_ms(); \
        long allocated_bytes = (long)(out_c) * (out_h) * (out_w) * sizeof(uint8_t); \
        strncpy(profiles[num_profiles].name, layer_name, 31); \
        profiles[num_profiles].name[31] = '\0'; \
        profiles[num_profiles].time_ms = layer_end - layer_start; \
        profiles[num_profiles].alloc_bytes = allocated_bytes; \
        num_profiles++; \
    } \
} while(0)

// --- STATIC BUFFERS ---
static uint8_t BUFFER_A[MAX_FEATURES];
static uint8_t BUFFER_B[MAX_FEATURES];
static uint8_t BUFFER_RESIDUAL[MAX_FEATURES];

// Weights Memory Pool
static int8_t FLASH_ROM[MAX_ROM_SIZE];
static size_t rom_head = 0; 

// Biases Memory Pool
static float FLASH_ROM_BIAS[MAX_FEATURES]; 
static size_t bias_head = 0;

int get_max_index(float *logits, int num_classes) {
    int max_idx = 0;
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > logits[max_idx]) max_idx = i;
    }
    return max_idx;
}

// --- BARE METAL PARSERS ---
QTensor load_qtensor(FILE *fp, int num_elements) {
    QTensor qt;
    if (fread(&qt.scale, sizeof(float), 1, fp) != 1) printf("❌ Error reading scale\n");
    if (fread(&qt.zero_point, sizeof(int32_t), 1, fp) != 1) printf("❌ Error reading ZP\n");
    
    qt.weights = &FLASH_ROM[rom_head];
    if (fread(&FLASH_ROM[rom_head], sizeof(int8_t), num_elements, fp) != (size_t)num_elements) {
        printf("❌ Error reading INT8 weights\n");
    }
    rom_head += num_elements;

    if (fread(&qt.out_scale, sizeof(float), 1, fp) != 1) printf("❌ Error reading out_scale\n");
    if (fread(&qt.out_zero_point, sizeof(int32_t), 1, fp) != 1) printf("❌ Error reading out_ZP\n");

    // UNIVERSAL BIAS LOADER
    int out_c;
    if (fread(&out_c, sizeof(int), 1, fp) != 1) printf("❌ Error reading bias count\n");
    
    // 🛑 THE CRASH PREVENTER
    if (out_c <= 0 || out_c > 100000) {
        printf("\n💥 FATAL STREAM DESYNC!\n");
        printf("The C engine tried to allocate %d biases at ROM head %zu.\n", out_c, rom_head);
        printf("This means the Python binary export is misaligned. Exiting gracefully.\n");
        exit(1);
    }
    
    qt.bias = &FLASH_ROM_BIAS[bias_head];
    if (fread(&FLASH_ROM_BIAS[bias_head], sizeof(float), out_c, fp) != (size_t)out_c) {
        printf("❌ Error reading biases\n");
    }
    bias_head += out_c;

    return qt;
}

uint8_t* run_mbconv_block_int8(uint8_t *curr_in, uint8_t *curr_out, uint8_t *res_buf,
                              int in_c, int expand_ratio, int out_c, int h, int w, int stride, int k,
                              QTensor exp_q, QTensor dw_q, 
                              QTensor se_1_q, QTensor se_3_q, 
                              QTensor proj_q,
                              float s_in, int32_t z_in, 
                              int profile_flag) 
{
    uint8_t *A = curr_in, *B = curr_out, *temp;
    int hidden_c = in_c * expand_ratio;
    int use_res = (in_c == out_c && stride == 1);
    
    float orig_s_in = s_in;
    int32_t orig_z_in = z_in;

    if (use_res) { for(int i = 0; i < in_c * h * w; i++) res_buf[i] = A[i]; }

    if (expand_ratio != 1) {
        // FIXED: Replaced NULL with exp_q.bias
        conv2d_int8(A, B, exp_q.weights, exp_q.bias, in_c, hidden_c, h, w, 1, 1, 
                    s_in, z_in, exp_q.scale, exp_q.zero_point, exp_q.out_scale, exp_q.out_zero_point);
        apply_silu_int8(B, hidden_c * h * w, exp_q.out_scale, exp_q.out_zero_point, exp_q.out_scale, exp_q.out_zero_point);
        s_in = exp_q.out_scale; z_in = exp_q.out_zero_point;
        temp = A; A = B; B = temp;
    }

    int next_h = (h + 2*(k/2) - k) / stride + 1;
    int next_w = (w + 2*(k/2) - k) / stride + 1;
    
    // FIXED: Replaced NULL with dw_q.bias
    conv2d_depthwise_int8(A, B, dw_q.weights, dw_q.bias, hidden_c, h, w, k, stride,
                          s_in, z_in, dw_q.scale, dw_q.zero_point, dw_q.out_scale, dw_q.out_zero_point);
    apply_silu_int8(B, hidden_c * next_h * next_w, dw_q.out_scale, dw_q.out_zero_point, dw_q.out_scale, dw_q.out_zero_point);
    s_in = dw_q.out_scale; z_in = dw_q.out_zero_point;
    temp = A; A = B; B = temp;

    int reduced_c = in_c / 4;
    if (reduced_c < 1) reduced_c = 1;
    
    // FIXED: Passing biases directly from the structs
    squeeze_excitation_int8(A, hidden_c, next_h, next_w, reduced_c,
                            se_1_q.weights, se_1_q.bias, se_1_q.scale, se_1_q.zero_point,
                            se_3_q.weights, se_3_q.bias, se_3_q.scale, se_3_q.zero_point,
                            s_in, z_in, s_in, z_in);

    // FIXED: Replaced NULL with proj_q.bias
    conv2d_int8(A, B, proj_q.weights, proj_q.bias, hidden_c, out_c, next_h, next_w, 1, 1,
                s_in, z_in, proj_q.scale, proj_q.zero_point, proj_q.out_scale, proj_q.out_zero_point);
    temp = A; A = B; B = temp;

    if (use_res) { 
        skip_add_int8(A, res_buf, out_c * next_h * next_w, 
                      proj_q.out_scale, proj_q.out_zero_point, 
                      orig_s_in, orig_z_in, 
                      proj_q.out_scale, proj_q.out_zero_point); 
    }
    return A;
}

int main() {
    printf("=================================================================================\n");
    printf("🚀 INT8 HARDWARE INFERENCE PROFILING (Golden Oracle Validation)\n");
    printf("=================================================================================\n\n");

    FILE *fp_d = fopen("mnist_test_data.bin", "rb");
    FILE *fp_py = fopen("pytorch_preds.bin", "rb");
    FILE *fp_w = fopen("quantized_weights.bin", "rb");

    if (!fp_d || !fp_py || !fp_w) {
        printf("❌ ERROR: Missing one of the required .bin files!\n");
        return 1;
    }

    printf("Loading INT8 model into static ROM buffer...\n");
    
    // --- STATIC ROM ALLOCATION ---
    QTensor stem_conv = load_qtensor(fp_w, 32 * 3 * 3 * 3);
    
    // MBConv Block 1
    QTensor b1_dw = load_qtensor(fp_w, 32 * 3 * 3);
    QTensor b1_se1 = load_qtensor(fp_w, 8 * 32); 
    QTensor b1_se3 = load_qtensor(fp_w, 32 * 8); 
    QTensor b1_proj = load_qtensor(fp_w, 16 * 32 * 1 * 1);

    // MBConv Block 2
    QTensor b2_exp = load_qtensor(fp_w, 96 * 16 * 1 * 1);
    QTensor b2_dw = load_qtensor(fp_w, 96 * 3 * 3);
    QTensor b2_se1 = load_qtensor(fp_w, 4 * 96); 
    QTensor b2_se3 = load_qtensor(fp_w, 96 * 4); 
    QTensor b2_proj = load_qtensor(fp_w, 24 * 96 * 1 * 1);
    
    // MBConv Block 3 (in:24, out:24, exp:6, k:3) | hidden=144, reduced=6
    QTensor b3_exp = load_qtensor(fp_w, 144 * 24);
    QTensor b3_dw = load_qtensor(fp_w, 144 * 9);
    QTensor b3_se1 = load_qtensor(fp_w, 6 * 144); 
    QTensor b3_se3 = load_qtensor(fp_w, 144 * 6); 
    QTensor b3_proj = load_qtensor(fp_w, 24 * 144);

    // MBConv Block 4 (in:24, out:40, exp:6, k:5) | hidden=144, reduced=6
    QTensor b4_exp = load_qtensor(fp_w, 144 * 24);
    QTensor b4_dw = load_qtensor(fp_w, 144 * 25);
    QTensor b4_se1 = load_qtensor(fp_w, 6 * 144); 
    QTensor b4_se3 = load_qtensor(fp_w, 144 * 6); 
    QTensor b4_proj = load_qtensor(fp_w, 40 * 144);

    // MBConv Block 5 (in:40, out:40, exp:6, k:5) | hidden=240, reduced=10
    QTensor b5_exp = load_qtensor(fp_w, 240 * 40);
    QTensor b5_dw = load_qtensor(fp_w, 240 * 25);
    QTensor b5_se1 = load_qtensor(fp_w, 10 * 240); 
    QTensor b5_se3 = load_qtensor(fp_w, 240 * 10); 
    QTensor b5_proj = load_qtensor(fp_w, 40 * 240);

    // MBConv Block 6 (in:40, out:80, exp:6, k:3) | hidden=240, reduced=10
    QTensor b6_exp = load_qtensor(fp_w, 240 * 40);
    QTensor b6_dw = load_qtensor(fp_w, 240 * 9);
    QTensor b6_se1 = load_qtensor(fp_w, 10 * 240); 
    QTensor b6_se3 = load_qtensor(fp_w, 240 * 10); 
    QTensor b6_proj = load_qtensor(fp_w, 80 * 240);

    // MBConv Block 7 (in:80, out:80, exp:6, k:3) | hidden=480, reduced=20
    QTensor b7_exp = load_qtensor(fp_w, 480 * 80);
    QTensor b7_dw = load_qtensor(fp_w, 480 * 9);
    QTensor b7_se1 = load_qtensor(fp_w, 20 * 480); 
    QTensor b7_se3 = load_qtensor(fp_w, 480 * 20);
    QTensor b7_proj = load_qtensor(fp_w, 80 * 480);

    // MBConv Block 8 (in:80, out:80, exp:6, k:3) | hidden=480, reduced=20
    QTensor b8_exp = load_qtensor(fp_w, 480 * 80);
    QTensor b8_dw = load_qtensor(fp_w, 480 * 9);
    QTensor b8_se1 = load_qtensor(fp_w, 20 * 480); 
    QTensor b8_se3 = load_qtensor(fp_w, 480 * 20); 
    QTensor b8_proj = load_qtensor(fp_w, 80 * 480);

    // MBConv Block 9 (in:80, out:112, exp:6, k:5) | hidden=480, reduced=20
    QTensor b9_exp = load_qtensor(fp_w, 480 * 80);
    QTensor b9_dw = load_qtensor(fp_w, 480 * 25);
    QTensor b9_se1 = load_qtensor(fp_w, 20 * 480); 
    QTensor b9_se3 = load_qtensor(fp_w, 480 * 20); 
    QTensor b9_proj = load_qtensor(fp_w, 112 * 480);

    // MBConv Block 10 (in:112, out:112, exp:6, k:5) | hidden=672, reduced=28
    QTensor b10_exp = load_qtensor(fp_w, 672 * 112);
    QTensor b10_dw = load_qtensor(fp_w, 672 * 25);
    QTensor b10_se1 = load_qtensor(fp_w, 28 * 672); 
    QTensor b10_se3 = load_qtensor(fp_w, 672 * 28); 
    QTensor b10_proj = load_qtensor(fp_w, 112 * 672);

    // MBConv Block 11 (in:112, out:112, exp:6, k:5) | hidden=672, reduced=28
    QTensor b11_exp = load_qtensor(fp_w, 672 * 112);
    QTensor b11_dw = load_qtensor(fp_w, 672 * 25);
    QTensor b11_se1 = load_qtensor(fp_w, 28 * 672); 
    QTensor b11_se3 = load_qtensor(fp_w, 672 * 28); 
    QTensor b11_proj = load_qtensor(fp_w, 112 * 672);

    // MBConv Block 12 (in:112, out:192, exp:6, k:5) | hidden=672, reduced=28
    QTensor b12_exp = load_qtensor(fp_w, 672 * 112);
    QTensor b12_dw = load_qtensor(fp_w, 672 * 25);
    QTensor b12_se1 = load_qtensor(fp_w, 28 * 672); 
    QTensor b12_se3 = load_qtensor(fp_w, 672 * 28); 
    QTensor b12_proj = load_qtensor(fp_w, 192 * 672);

    // MBConv Block 13 (in:192, out:192, exp:6, k:5) | hidden=1152, reduced=48
    QTensor b13_exp = load_qtensor(fp_w, 1152 * 192);
    QTensor b13_dw = load_qtensor(fp_w, 1152 * 25);
    QTensor b13_se1 = load_qtensor(fp_w, 48 * 1152); 
    QTensor b13_se3 = load_qtensor(fp_w, 1152 * 48); 
    QTensor b13_proj = load_qtensor(fp_w, 192 * 1152);

    // MBConv Block 14 (in:192, out:192, exp:6, k:5) | hidden=1152, reduced=48
    QTensor b14_exp = load_qtensor(fp_w, 1152 * 192);
    QTensor b14_dw = load_qtensor(fp_w, 1152 * 25);
    QTensor b14_se1 = load_qtensor(fp_w, 48 * 1152); 
    QTensor b14_se3 = load_qtensor(fp_w, 1152 * 48); 
    QTensor b14_proj = load_qtensor(fp_w, 192 * 1152);

    // MBConv Block 15 (in:192, out:192, exp:6, k:5) | hidden=1152, reduced=48
    QTensor b15_exp = load_qtensor(fp_w, 1152 * 192);
    QTensor b15_dw = load_qtensor(fp_w, 1152 * 25);
    QTensor b15_se1 = load_qtensor(fp_w, 48 * 1152); 
    QTensor b15_se3 = load_qtensor(fp_w, 1152 * 48); 
    QTensor b15_proj = load_qtensor(fp_w, 192 * 1152);

    // MBConv Block 16 (in:192, out:320, exp:6, k:3) | hidden=1152, reduced=48
    QTensor b16_exp = load_qtensor(fp_w, 1152 * 192);
    QTensor b16_dw = load_qtensor(fp_w, 1152 * 9);
    QTensor b16_se1 = load_qtensor(fp_w, 48 * 1152); 
    QTensor b16_se3 = load_qtensor(fp_w, 1152 * 48); 
    QTensor b16_proj = load_qtensor(fp_w, 320 * 1152);
    
    // Head & Classifier
    QTensor head_conv = load_qtensor(fp_w, 1280 * 320 * 1 * 1);
    QTensor class_wt = load_qtensor(fp_w, 10 * 1280); 
    fclose(fp_w);
    printf("✅ Model loaded! Total ROM consumed: %zu bytes (%.2f KB)\n\n", rom_head, rom_head / 1024.0);

    int correct_predictions = 0, match_count = 0;
    int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};
    float total_loss = 0.0f;
    double start_time = get_time_ms();

    for (int img_idx = 0; img_idx < NUM_TEST_IMAGES; img_idx++) {
        float input_img[3072];
        int current_label, py_pred; 
        
        if (fread(&current_label, sizeof(int), 1, fp_d) != 1) break;
        if (fread(input_img, sizeof(float), 3072, fp_d) != 3072) break;
        if (fread(&py_pred, sizeof(int), 1, fp_py) != 1) break;

        double layer_start = 0, layer_end = 0;
        void *heap_before = NULL;
        
        if (img_idx == 0) {
            heap_before = sbrk(0);
        }

        // ✅ FIXED: Pointers are now explicitly uint8_t
        uint8_t *in_buf = BUFFER_A;
        uint8_t *out_buf = BUFFER_B;
        int profile_flag = (img_idx == 0);

        // --- ENTER INT8 REALM ---
        // Converts the 3072 floats into 3072 scaled integers
        quantize_tensor(input_img, in_buf, 3072, 0.00783f, 128); 

        // --- 1. STEM ---
        PROFILE_START();
        double h_start = profile_flag ? get_time_ms() : 0;
        
        // Raw image pixels enter with a scale of 0.0078f (1/128)
        conv2d_int8(in_buf, out_buf, stem_conv.weights, stem_conv.bias, 3, 32, 32, 32, 3, 2, 
                    0.00783f, 128, stem_conv.scale, stem_conv.zero_point, stem_conv.out_scale, stem_conv.out_zero_point);
        apply_silu_int8(out_buf, 32 * 16 * 16, stem_conv.out_scale, stem_conv.out_zero_point, stem_conv.out_scale, stem_conv.out_zero_point);
        
        // 🔄 DYNAMIC TRACKER: Capture the exact scale flowing out of the stem
        float current_scale = stem_conv.out_scale;
        int32_t current_zp = stem_conv.out_zero_point;
        
        acc_pw_ms += (get_time_ms() - h_start);
        in_buf = out_buf; out_buf = BUFFER_A;
        PROFILE_END("Stem", 32, 16, 16);


        // --- 2. MBCONV BLOCKS ---

        // Block 1 (in:32, out:16, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 32, 1, 16, 16, 16, 1, 3, 
                                       (QTensor){}, b1_dw, b1_se1, b1_se3, b1_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b1_proj.out_scale; current_zp = b1_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 1", 16, 16, 16);

        // Block 2 (in:16, out:24, stride:2) -> Drops to 8x8
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 16, 6, 24, 16, 16, 2, 3, 
                                       b2_exp, b2_dw, b2_se1, b2_se3, b2_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b2_proj.out_scale; current_zp = b2_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 2", 24, 8, 8);

        // Block 3 (in:24, out:24, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 24, 6, 24, 8, 8, 1, 3, 
                                       b3_exp, b3_dw, b3_se1, b3_se3, b3_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b3_proj.out_scale; current_zp = b3_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 3", 24, 8, 8);

        // Block 4 (in:24, out:40, stride:2) -> Drops to 4x4
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 24, 6, 40, 8, 8, 2, 5, 
                                       b4_exp, b4_dw, b4_se1, b4_se3, b4_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b4_proj.out_scale; current_zp = b4_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 4", 40, 4, 4);

        // Block 5 (in:40, out:40, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 40, 6, 40, 4, 4, 1, 5, 
                                       b5_exp, b5_dw, b5_se1, b5_se3, b5_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b5_proj.out_scale; current_zp = b5_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 5", 40, 4, 4);

        // Block 6 (in:40, out:80, stride:2) -> Drops to 2x2
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 40, 6, 80, 4, 4, 2, 3, 
                                       b6_exp, b6_dw, b6_se1, b6_se3, b6_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b6_proj.out_scale; current_zp = b6_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 6", 80, 2, 2);

        // Block 7 (in:80, out:80, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 80, 6, 80, 2, 2, 1, 3, 
                                       b7_exp, b7_dw, b7_se1, b7_se3, b7_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b7_proj.out_scale; current_zp = b7_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 7", 80, 2, 2);

        // Block 8 (in:80, out:80, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 80, 6, 80, 2, 2, 1, 3, 
                                       b8_exp, b8_dw, b8_se1, b8_se3, b8_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b8_proj.out_scale; current_zp = b8_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 8", 80, 2, 2);

        // Block 9 (in:80, out:112, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 80, 6, 112, 2, 2, 1, 5, 
                                       b9_exp, b9_dw, b9_se1, b9_se3, b9_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b9_proj.out_scale; current_zp = b9_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 9", 112, 2, 2);

        // Block 10 (in:112, out:112, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 112, 6, 112, 2, 2, 1, 5, 
                                       b10_exp, b10_dw, b10_se1, b10_se3, b10_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b10_proj.out_scale; current_zp = b10_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 10", 112, 2, 2);

        // Block 11 (in:112, out:112, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 112, 6, 112, 2, 2, 1, 5, 
                                       b11_exp, b11_dw, b11_se1, b11_se3, b11_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b11_proj.out_scale; current_zp = b11_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 11", 112, 2, 2);

        // Block 12 (in:112, out:192, stride:2) -> Drops to 1x1
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 112, 6, 192, 2, 2, 2, 5, 
                                       b12_exp, b12_dw, b12_se1, b12_se3, b12_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b12_proj.out_scale; current_zp = b12_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 12", 192, 1, 1);

        // Block 13 (in:192, out:192, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 192, 6, 192, 1, 1, 1, 5, 
                                       b13_exp, b13_dw, b13_se1, b13_se3, b13_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b13_proj.out_scale; current_zp = b13_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 13", 192, 1, 1);

        // Block 14 (in:192, out:192, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 192, 6, 192, 1, 1, 1, 5, 
                                       b14_exp, b14_dw, b14_se1, b14_se3, b14_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b14_proj.out_scale; current_zp = b14_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 14", 192, 1, 1);

        // Block 15 (in:192, out:192, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 192, 6, 192, 1, 1, 1, 5, 
                                       b15_exp, b15_dw, b15_se1, b15_se3, b15_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b15_proj.out_scale; current_zp = b15_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 15", 192, 1, 1);

        // Block 16 (in:192, out:320, stride:1)
        PROFILE_START();
        in_buf = run_mbconv_block_int8(in_buf, out_buf, BUFFER_RESIDUAL, 192, 6, 320, 1, 1, 1, 3, 
                                       b16_exp, b16_dw, b16_se1, b16_se3, b16_proj,
                                       current_scale, current_zp, profile_flag);
        current_scale = b16_proj.out_scale; current_zp = b16_proj.out_zero_point;
        out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("MBConv Block 16", 320, 1, 1);

        // --- 3. HEAD ---
        PROFILE_START();
        
        h_start = profile_flag ? get_time_ms() : 0;
        
        conv2d_int8(in_buf, out_buf, head_conv.weights, head_conv.bias, 320, 1280, 1, 1, 1, 1,
                    current_scale, current_zp, head_conv.scale, head_conv.zero_point, head_conv.out_scale, head_conv.out_zero_point);
        
        apply_silu_int8(out_buf, 1280, head_conv.out_scale, head_conv.out_zero_point, head_conv.out_scale, head_conv.out_zero_point);
        
        // Final handoff to the Pool/Linear classifier
        current_scale = head_conv.out_scale;
        current_zp = head_conv.out_zero_point;
        
        acc_pw_ms += (get_time_ms() - h_start);     
        in_buf = out_buf; out_buf = (in_buf == BUFFER_A) ? BUFFER_B : BUFFER_A;
        PROFILE_END("Head", 1280, 1, 1);

        // --- 4. POOL & CLASSIFY ---
        PROFILE_START();
        double p_start = profile_flag ? get_time_ms() : 0;
        
        // DEQUANTIZE: Convert the integers back to usable float decimals using the dynamic scale!
        float float_pool[1280] = {0};
        for (int c = 0; c < 1280; c++) {
            float_pool[c] = ((float)in_buf[c] - current_zp) * current_scale;
        }

        float final_logits[10] = {0};
        for (int c = 0; c < 10; c++) {
            float acc = 0;
            for (int i = 0; i < 1280; i++) {
                acc += float_pool[i] * ((class_wt.weights[c * 1280 + i] - class_wt.zero_point) * class_wt.scale);
            }
            final_logits[c] = acc + class_wt.bias[c];
        }

        acc_pool_fc_ms += (get_time_ms() - p_start);
        PROFILE_END("Pool & Classify", 10, 1, 1);

        // --- BARE-METAL PROFILER END ---
        if (img_idx == 0) {
            void *heap_after = sbrk(0);
            global_heap_diff = (long)((char*)heap_after - (char*)heap_before);

        }

        // --- 5. METRICS & LIVE ORACLE MATCHING ---
        // (Uses final_logits array now since we are in the float realm)
        int c_pred = get_max_index(final_logits, 10);
        
        if (c_pred != py_pred) {
            printf("❌ MISMATCH at Index [%4d] | True Label: %d | PyTorch Predicted: %d | C Predicted: %d\n", 
                   img_idx, current_label, py_pred, c_pred);
        } else {
            match_count++; 
        }
        
        if (current_label >= 0 && current_label < 10 && c_pred >= 0 && c_pred < 10) {
            confusion_matrix[current_label][c_pred]++;
            if (c_pred == current_label) correct_predictions++;
        }
        
        // ADD THIS LINE: Accumulate the loss to measure model confidence
        total_loss += cross_entropy_loss(final_logits, current_label, 10);
        
        if ((img_idx + 1) % 2000 == 0) {
            printf("Processed %d / %d images...\n", img_idx + 1, NUM_TEST_IMAGES);
        }
    }
    
    fclose(fp_d);
    fclose(fp_py); 

    // --- 6. FINAL ENGINEERING REPORT ---
    double total_eval_time = get_time_ms() - start_time;
    float avg_latency = total_eval_time / NUM_TEST_IMAGES;
    float fps = 1000.0f / avg_latency;
    float avg_loss = total_loss / NUM_TEST_IMAGES;

    float accuracy = (float)correct_predictions / NUM_TEST_IMAGES * 100.0f;
    float match_rate = (float)match_count / NUM_TEST_IMAGES * 100.0f;

    long static_ram_bytes = sizeof(BUFFER_A) + sizeof(BUFFER_B) + sizeof(BUFFER_RESIDUAL);

    printf("\n=================================================================================\n");
    printf(" 🚀 BARE-METAL INT8 INFERENCE ENGINE : FINAL TELEMETRY REPORT \n");
    printf("=================================================================================\n");
    printf(" [1] ACCURACY & VALIDATION\n");
    printf("  ├─ Total Images Processed : %d\n", NUM_TEST_IMAGES);
    printf("  ├─ Golden Oracle Match    : %.2f%% (%d / %d)\n", match_rate, match_count, NUM_TEST_IMAGES);
    printf("  ├─ Hardware INT8 Accuracy : %.2f%% (%d / %d)\n", accuracy, correct_predictions, NUM_TEST_IMAGES);
    printf("  └─ Average Cross-Entropy  : %.4f (Confidence Score)\n", avg_loss);
    printf("---------------------------------------------------------------------------------\n");
    
    printf(" [2] HARDWARE PERFORMANCE & THROUGHPUT\n");
    printf("  ├─ Total Evaluation Time  : %.2f Seconds\n", total_eval_time / 1000.0);
    printf("  ├─ Average Latency        : %.4f ms / image\n", avg_latency);
    printf("  └─ Estimated Throughput   : %.2f Frames Per Second (FPS)\n", fps);
    printf("\n  [CPU Bottleneck Breakdown]\n");
    printf("     - Pointwise / Linear   : %.4f ms (%.1f%%)\n", acc_pw_ms/NUM_TEST_IMAGES, (acc_pw_ms/total_eval_time)*100);
    printf("     - Pooling & FC         : %.4f ms (%.1f%%)\n", acc_pool_fc_ms/NUM_TEST_IMAGES, (acc_pool_fc_ms/total_eval_time)*100);
    printf("---------------------------------------------------------------------------------\n");

    printf(" [3] STATIC MEMORY FOOTPRINT\n");
    printf("  ├─ Read-Only Memory (ROM) : %.2f KB (Weights, Scales, Biases)\n", rom_head / 1024.0);
    printf("  └─ Random Access RAM (BSS): %.2f KB (Ping-Pong Buffers)\n", static_ram_bytes / 1024.0);
    printf("---------------------------------------------------------------------------------\n\n");

    printf(" 🧮 CONFUSION MATRIX (True Labels vs. C Engine Predictions)\n");
    printf("      ");
    for (int i = 0; i < NUM_CLASSES; i++) printf("P%-3d ", i);
    printf("\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("T%-3d |", i);
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (i == j) {
                printf("\033[1;32m%4d\033[0m ", confusion_matrix[i][j]); 
            } else if (confusion_matrix[i][j] > 0) {
                printf("\033[1;31m%4d\033[0m ", confusion_matrix[i][j]); 
            } else {
                printf("%4d ", confusion_matrix[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n---------------------------------------------------------------------------------\n\n");

    printf(" 📈 PER-CLASS METRICS\n");
    printf("| Class | Precision |  Recall  | F1-Score |\n");
    printf("|-------|-----------|----------|----------|\n");
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        int tp = confusion_matrix[i][i];
        int fp = 0, fn = 0;
        
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (i != j) {
                fp += confusion_matrix[j][i]; 
                fn += confusion_matrix[i][j]; 
            }
        }
        
        float precision = (tp + fp) == 0 ? 0 : (float)tp / (tp + fp);
        float recall = (tp + fn) == 0 ? 0 : (float)tp / (tp + fn);
        float f1 = (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
        
        printf("|  %2d   |  %6.2f%%  | %6.2f%%  |  %6.2f  |\n", 
               i, precision * 100.0f, recall * 100.0f, f1 * 100.0f);
    }
    printf("=================================================================================\n\n");

    printf(" [4] LAYER-WISE LATENCY & MEMORY MAP (PING-PONG SRAM)\n");
    printf("| %-20s | %13s | %12s | %12s |\n", "Neural Block / Layer", "Time (ms)", "Alloc (Bytes)", "Alloc (KB)");
    printf("|----------------------|---------------|--------------|--------------|\n");
    
    long max_layer_ram = 0;
    for (int i = 0; i < num_profiles; i++) {
        if (profiles[i].alloc_bytes > max_layer_ram) {
            max_layer_ram = profiles[i].alloc_bytes;
        }
        printf("| %-20s | %10.4f ms | %12ld | %9.2f KB |\n", 
               profiles[i].name, 
               profiles[i].time_ms, 
               profiles[i].alloc_bytes, 
               profiles[i].alloc_bytes / 1024.0);
    }
    printf("---------------------------------------------------------------------------------\n");
    
    printf(" [5] MICROCONTROLLER DEPLOYMENT VIABILITY\n");
    printf("  ├─ Global Max Reserved RAM (.bss): %.2f KB\n", static_ram_bytes / 1024.0);
    printf("  ├─ Peak Layer Execution RAM      : %.2f KB\n", max_layer_ram / 1024.0);
    printf("  ├─ Heap Leaked/Changed           : %ld bytes\n", global_heap_diff);
    
    if (global_heap_diff == 0 && (static_ram_bytes / 1024.0) < 512.0) {
        printf("  └─ Status                        : \033[1;32mREADY FOR HARDWARE DEPLOYMENT\033[0m\n");
    } else {
        printf("  └─ Status                        : \033[1;31mMEMORY LEAK OR OVERFLOW DETECTED\033[0m\n");
    }
    printf("=================================================================================\n\n");

    return 0;
}