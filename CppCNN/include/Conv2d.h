#pragma once
#include <cmath>
#include <iostream>
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"

using namespace std;

class Conv2d : public Layer {
private:
    Matrix kernel;
    Matrix bias;
    Matrix padded_input_cache;
    std::size_t in_channels;
    std::size_t out_channels;
    std::size_t kernel_size;
    std::size_t stride;
    std::size_t padding;

    // Adam optimizer parameters
    Matrix m_kernel;  // First moment for kernel
    Matrix v_kernel;  // Second moment for kernel
    Matrix m_bias;    // First moment for bias
    Matrix v_bias;    // Second moment for bias
    size_t t;        // Time step counter
    float beta1;     // Exponential decay rate for first moment
    float beta2;     // Exponential decay rate for second moment
    float epsilon;   // Small constant for numerical stability

public:
    Conv2d(std::size_t in_channels, std::size_t out_channels,
        std::size_t kernel_size, std::size_t stride = 1,
        std::size_t padding = 0)
        : in_channels(in_channels)
        , out_channels(out_channels)
        , kernel_size(kernel_size)
        , stride(stride)
        , padding(padding)
        , kernel(out_channels, in_channels, kernel_size, kernel_size)
        , bias(out_channels, 1, 1, 1)
        , padded_input_cache(1, 1, 1, 1)

        , m_kernel(out_channels, in_channels, kernel_size, kernel_size)
        , v_kernel(out_channels, in_channels, kernel_size, kernel_size)
        , m_bias(out_channels, 1, 1, 1)
        , v_bias(out_channels, 1, 1, 1)
        , t(0)
        , beta1(0.9f)
        , beta2(0.999f)
        , epsilon(1e-8f)
    {
        float limit = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
        limit *= std::sqrt(2.0f);
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t ic = 0; ic < in_channels; ++ic) {
                for (std::size_t h = 0; h < kernel_size; ++h) {
                    for (std::size_t w = 0; w < kernel_size; ++w) {
                        float random_val = ((float)rand() / RAND_MAX) * 2 * limit - limit;
                        kernel.at(oc, ic, h, w) = random_val;

                        m_kernel.at(oc, ic, h, w) = 0.0f;
                        v_kernel.at(oc, ic, h, w) = 0.0f;
                    }
                }
            }
            bias.at(oc, 0, 0, 0) = 0.0f;
            m_bias.at(oc, 0, 0, 0) = 0.0f;
            v_bias.at(oc, 0, 0, 0) = 0.0f;
        }
    }

    std::pair<std::size_t, std::size_t> get_output_dims(std::size_t H, std::size_t W) const {
        std::size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
        std::size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
        return { H_out, W_out };
    }

    Matrix forward(const Matrix& input) override {
        std::size_t B = input.batch_size();
        std::size_t C = input.channels();
        std::size_t H = input.height();
        std::size_t W = input.width();

        if (B == 0 || C != in_channels || H == 0 || W == 0) {
            throw std::invalid_argument("Invalid input dimensions");
        }

        auto [H_out, W_out] = get_output_dims(H, W);
        if (H_out == 0 || W_out == 0) {
            throw std::invalid_argument("Invalid output dimensions");
        }

        Matrix padded_input(B, C, H + 2 * padding, W + 2 * padding);
        Matrix output(B, out_channels, H_out, W_out);

#pragma omp parallel for collapse(2)
        for (size_t b = 0; b < B; b++) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                const float bias_val = bias.at(oc, 0, 0, 0);
                for (size_t oh = 0; oh < H_out; oh++) {
#pragma omp simd
                    for (size_t ow = 0; ow < W_out; ow++) {
                        output.at(b, oc, oh, ow) = bias_val;
                    }
                }
            }
        }

        if (padding > 0) {
#pragma omp parallel for collapse(3)
            for (size_t b = 0; b < B; b++) {
                for (size_t c = 0; c < C; c++) {
                    for (size_t h = 0; h < H; h++) {
#pragma omp simd
                        for (size_t w = 0; w < W; w++) {
                            padded_input.at(b, c, h + padding, w + padding) = input.at(b, c, h, w);
                        }
                    }
                }
            }
        }

        const Matrix& conv_input = (padding > 0) ? padded_input : input;
        padded_input_cache = conv_input;

#pragma omp parallel for collapse(3)
        for (size_t b = 0; b < B; b++) {
            for (size_t oc = 0; oc < out_channels; oc++) {
                for (size_t oh = 0; oh < H_out; oh++) {
                    for (size_t ow = 0; ow < W_out; ow++) {
                        float sum = 0.0f;

                        for (size_t ic = 0; ic < in_channels; ic++) {
                            size_t ih_start = oh * stride;
                            size_t iw_start = ow * stride;

                            for (size_t kh = 0; kh < kernel_size; kh++) {
                                const size_t ih = ih_start + kh;
#pragma omp simd reduction(+:sum)
                                for (size_t kw = 0; kw < kernel_size; kw++) {
                                    const size_t iw = iw_start + kw;
                                    sum += conv_input.at(b, ic, ih, iw) *
                                        kernel.at(oc, ic, kh, kw);
                                }
                            }
                        }
                        output.at(b, oc, oh, ow) += sum;
                    }
                }
            }
        }

        return output;
    }
    //
    //    Matrix forward(const Matrix& input) override {
    //        std::size_t B = input.batch_size();
    //        std::size_t C = input.channels();
    //        std::size_t H = input.height();
    //        std::size_t W = input.width();
    //
    //        if (B == 0 || C != in_channels || H == 0 || W == 0) {
    //            throw std::invalid_argument("Invalid input dimensions");
    //        }
    //        auto [H_out, W_out] = get_output_dims(H, W);
    //        if (H_out == 0 || W_out == 0) {
    //            throw std::invalid_argument("Invalid output dimensions");
    //        }
    //
    //        Matrix padded_input(B, C, H + 2 * padding, W + 2 * padding);
    //        Matrix output(B, out_channels, H_out, W_out);
    //
    //        // copying and padding input
    //#pragma omp parallel for collapse(4)
    //        for (size_t b = 0; b < B; b++) {
    //            for (size_t c = 0; c < C; c++) {
    //                for (size_t h = 0; h < H; h++) {
    //                    for (size_t w = 0; w < W; w++) {
    //                        padded_input.at(b, c, h + padding, w + padding) = input.at(b, c, h, w);
    //                    }
    //                }
    //            }
    //        }
    //        padded_input_cache = padded_input;
    //
    //        // 배치에 대해 output 생성
    //#pragma omp parallel for collapse(2)
    //        for (size_t b = 0; b < B; b++) {
    //
    //            // oc와 ic들에 대해 output 생성
    //            for (size_t oc = 0; oc < out_channels; oc++) {
    //                for (size_t ic = 0; ic < in_channels; ic++) {
    //
    //                    // 하나의 채널에 대해 output 생성
    //                    for (size_t oh = 0; oh < H_out; oh++) {
    //                        for (size_t ow = 0; ow < W_out; ow++) {
    //
    //                            // output 한 칸당 커널 적용
    //                            for (size_t kh = 0; kh < kernel_size; kh++) {
    //                                for (size_t kw = 0; kw < kernel_size; kw++) {
    //                                    size_t ih = kh + oh * stride;
    //                                    size_t iw = kw + ow * stride;
    //
    //                                    if (ih >= padded_input.height() || iw >= padded_input.width()) {
    //                                        std::cout << "Index out of bounds: ih=" << ih << " iw=" << iw << " kh=" << kh << " oh=" << oh << std::endl;
    //                                        std::cout << "Input dims: H=" << padded_input.height() << " W=" << padded_input.width() << std::endl;
    //                                    }
    //
    //                                    output.at(b, oc, oh, ow) += padded_input.at(b, ic, ih, iw) * kernel.at(oc, ic, kh, kw);
    //                                }
    //                            }
    //                            output.at(b, oc, oh, ow) += bias.at(oc, 0, 0, 0);
    //                        }
    //                    }
    //
    //                }
    //            }
    //        }
    //        return output;
    //    }



    Matrix backward(const Matrix& gradient) override {
        std::size_t B = padded_input_cache.batch_size();
        std::size_t H = padded_input_cache.height();
        std::size_t W = padded_input_cache.width();
        std::size_t origH = H - 2 * padding;
        std::size_t origW = W - 2 * padding;
        auto [H_out, W_out] = get_output_dims(origH, origW);

        Matrix kernel_gradients(kernel.batch_size(), kernel.channels(), kernel.height(), kernel.width());
        Matrix bias_gradients(bias.batch_size(), bias.channels(), bias.height(), bias.width());
        Matrix tmp_input_gradients(B, in_channels, H, W);
        Matrix input_gradients(B, in_channels, origH, origW);

#pragma omp parallel for
        for (size_t oc = 0; oc < out_channels; oc++) {
            float bias_grad = 0.0f;
            for (size_t b = 0; b < B; b++) {
                for (size_t oh = 0; oh < H_out; oh++) {
#pragma omp simd reduction(+:bias_grad)
                    for (size_t ow = 0; ow < W_out; ow++) {
                        bias_grad += gradient.at(b, oc, oh, ow);
                    }
                }
            }
            bias.add_gradient(oc, 0, 0, 0, bias_grad);
        }

#pragma omp parallel for collapse(2)
        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t ic = 0; ic < in_channels; ic++) {
                for (size_t kh = 0; kh < kernel_size; kh++) {
                    for (size_t kw = 0; kw < kernel_size; kw++) {
                        float kernel_grad = 0.0f;

                        for (size_t b = 0; b < B; b++) {
                            for (size_t oh = 0; oh < H_out; oh++) {
                                for (size_t ow = 0; ow < W_out; ow++) {
                                    size_t ih = oh * stride + kh;
                                    size_t iw = ow * stride + kw;
                                    kernel_grad += gradient.at(b, oc, oh, ow) *
                                        padded_input_cache.at(b, ic, ih, iw);
                                }
                            }
                        }

                        kernel.add_gradient(oc, ic, kh, kw, kernel_grad);
                    }
                }

                for (size_t b = 0; b < B; b++) {
                    for (size_t oh = 0; oh < H_out; oh++) {
                        for (size_t ow = 0; ow < W_out; ow++) {
                            const float grad_val = gradient.at(b, oc, oh, ow);

                            for (size_t kh = 0; kh < kernel_size; kh++) {
                                size_t ih = oh * stride + kh;
#pragma omp simd
                                for (size_t kw = 0; kw < kernel_size; kw++) {
                                    size_t iw = ow * stride + kw;
#pragma omp atomic
                                    tmp_input_gradients.at(b, ic, ih, iw) +=
                                        grad_val * kernel.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for collapse(3)
        for (size_t b = 0; b < B; b++) {
            for (size_t c = 0; c < in_channels; c++) {
                for (size_t h = 0; h < origH; h++) {
#pragma omp simd
                    for (size_t w = 0; w < origW; w++) {
                        input_gradients.at(b, c, h, w) =
                            tmp_input_gradients.at(b, c, h + padding, w + padding);
                    }
                }
            }
        }

        return input_gradients;
    }



    //    Matrix backward(const Matrix& gradient) override {
    //        std::size_t B = padded_input_cache.batch_size();
    //        std::size_t C = padded_input_cache.channels();
    //        std::size_t H = padded_input_cache.height();
    //        std::size_t W = padded_input_cache.width();
    //        std::size_t origH = H - 2 * padding;
    //        std::size_t origW = W - 2 * padding;
    //
    //        Matrix tmp_input_gradients(B, C, H, W);
    //        Matrix input_gradients(B, C, origH, origW);
    //
    //        auto [H_out, W_out] = get_output_dims(origH, origW);
    //
    //#pragma omp parallel for
    //        for (size_t b = 0; b < B; b++) {
    //            for (size_t oc = 0; oc < out_channels; oc++) {
    //                for (size_t ic = 0; ic < in_channels; ic++) {
    //                    for (size_t oh = 0; oh < H_out; oh++) {
    //                        for (size_t ow = 0; ow < W_out; ow++) {
    //                            for (size_t kh = 0; kh < kernel_size; kh++) {
    //                                for (size_t kw = 0; kw < kernel_size; kw++) {
    //                                    size_t ih = oh * stride + kh;
    //                                    size_t iw = ow * stride + kw;
    //
    //                                    if (ih >= H || iw >= W) {
    //                                        cout << "[ERROR] out of range ih=" << ih << " iw=" << iw << " H=" << H << " W=" << W << " oh=" << oh << " ow=" << ow << endl;
    //                                    }
    //
    //                                    tmp_input_gradients.at(b, ic, ih, iw) += gradient.at(b, oc, oh, ow) * kernel.at(oc, ic, kh, kw);
    //                                    float grad = gradient.at(b, oc, oh, ow) * padded_input_cache.at(b, ic, ih, iw);
    //                                    
    //#pragma omp atomic
    //                                    kernel.add_gradient(oc, ic, kh, kw, grad);
    //                                }
    //                            }
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //
    //        for (size_t b = 0; b < B; b++) {
    //            for (size_t oc = 0; oc < out_channels; oc++) {
    //                for (size_t oh = 0; oh < H_out; oh++) {
    //                    for (size_t ow = 0; ow < W_out; ow++) {
    //                        float grad = gradient.at(b, oc, oh, ow);
    //                        bias.add_gradient(oc, 0, 0, 0, grad);
    //                    }
    //                }
    //            }
    //        }
    //
    //#pragma omp parallel for collapse(4)
    //        for (size_t b = 0; b < B; b++) {
    //            for (size_t c = 0; c < C; c++) {
    //                for (size_t h = 0; h < origH; h++) {
    //                    for (size_t w = 0; w < origW; w++) {
    //                        input_gradients.at(b, c, h, w) = tmp_input_gradients.at(b, c, h + padding, w + padding);
    //                    }
    //                }
    //            }
    //        }
    //
    //        return input_gradients;
    //    }

    void update_parameters(float learning_rate) override {
        t++; // Increment time step

        // Compute bias correction terms
        float correction1 = 1.0f / (1.0f - std::pow(beta1, t));
        float correction2 = 1.0f / (1.0f - std::pow(beta2, t));

        // Update kernel parameters
        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            for (std::size_t ic = 0; ic < in_channels; ++ic) {
                for (std::size_t kh = 0; kh < kernel_size; ++kh) {
                    for (std::size_t kw = 0; kw < kernel_size; ++kw) {
                        float grad = kernel.get_gradient(oc, ic, kh, kw);

                        // Update biased first moment estimate
                        m_kernel.at(oc, ic, kh, kw) = beta1 * m_kernel.at(oc, ic, kh, kw) +
                            (1 - beta1) * grad;

                        // Update biased second moment estimate
                        v_kernel.at(oc, ic, kh, kw) = beta2 * v_kernel.at(oc, ic, kh, kw) +
                            (1 - beta2) * grad * grad;

                        // Compute bias-corrected first and second moment estimates
                        float m_hat = m_kernel.at(oc, ic, kh, kw) * correction1;
                        float v_hat = v_kernel.at(oc, ic, kh, kw) * correction2;

                        // Update kernel weights
                        kernel.at(oc, ic, kh, kw) -= learning_rate * m_hat /
                            (std::sqrt(v_hat) + epsilon);
                    }
                }
            }

            // Update bias with Adam
            float bias_grad = bias.get_gradient(oc, 0, 0, 0);

            // Update biased first moment estimate for bias
            m_bias.at(oc, 0, 0, 0) = beta1 * m_bias.at(oc, 0, 0, 0) +
                (1 - beta1) * bias_grad;

            // Update biased second moment estimate for bias
            v_bias.at(oc, 0, 0, 0) = beta2 * v_bias.at(oc, 0, 0, 0) +
                (1 - beta2) * bias_grad * bias_grad;

            // Compute bias-corrected estimates for bias
            float m_hat_bias = m_bias.at(oc, 0, 0, 0) * correction1;
            float v_hat_bias = v_bias.at(oc, 0, 0, 0) * correction2;

            // Update bias
            bias.at(oc, 0, 0, 0) -= learning_rate * m_hat_bias /
                (std::sqrt(v_hat_bias) + epsilon);
        }

        kernel.zero_gradients();
        bias.zero_gradients();
    }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::Conv2d);
    }

    void saveParameters(std::ofstream& file) const override {
        file.write(reinterpret_cast<const char*>(&in_channels), sizeof(in_channels));
        file.write(reinterpret_cast<const char*>(&out_channels), sizeof(out_channels));
        file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
        file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
        file.write(reinterpret_cast<const char*>(&padding), sizeof(padding));

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        float weight = kernel.at(oc, ic, kh, kw);
                        file.write(reinterpret_cast<const char*>(&weight), sizeof(float));
                    }
                }
            }
        }

        for (size_t oc = 0; oc < out_channels; ++oc) {
            float bias_val = bias.at(oc, 0, 0, 0);
            file.write(reinterpret_cast<const char*>(&bias_val), sizeof(float));
        }
    }

    void loadParameters(std::ifstream& file) override {
        file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
        file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
        file.read(reinterpret_cast<char*>(&padding), sizeof(padding));

        kernel = Matrix(out_channels, in_channels, kernel_size, kernel_size);
        bias = Matrix(out_channels, 1, 1, 1);

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t kh = 0; kh < kernel_size; ++kh) {
                    for (size_t kw = 0; kw < kernel_size; ++kw) {
                        float weight;
                        file.read(reinterpret_cast<char*>(&weight), sizeof(float));
                        kernel.at(oc, ic, kh, kw) = weight;
                    }
                }
            }
        }

        for (size_t oc = 0; oc < out_channels; ++oc) {
            float bias_val;
            file.read(reinterpret_cast<char*>(&bias_val), sizeof(float));
            bias.at(oc, 0, 0, 0) = bias_val;
        }
    }

    std::size_t get_in_channels() const { return in_channels; }
    std::size_t get_out_channels() const { return out_channels; }
    std::size_t get_kernel_size() const { return kernel_size; }
    std::size_t get_stride() const { return stride; }
    std::size_t get_padding() const { return padding; }
    const Matrix& get_kernel() const { return kernel; }
    const Matrix& get_bias() const { return bias; }
    void set_kernel(const Matrix& k) { kernel = k; }
    void set_bias(const Matrix& b) { bias = b; }
};