#pragma once
#include <cmath>
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"

class Conv2d : public Layer {
private:
    Matrix kernel;
    Matrix bias;
    Matrix input_cache;
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
        , input_cache(1, 1, 1, 1)

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
        input_cache = input;

        std::size_t batch_size = input.batch_size();
        std::size_t H = input.height();
        std::size_t W = input.width();
        auto [H_out, W_out] = get_output_dims(H, W);
        Matrix output(batch_size, out_channels, H_out, W_out);

        for (std::size_t b = 0; b < batch_size; ++b) {
            for (std::size_t oc = 0; oc < out_channels; ++oc) {
                for (std::size_t oh = 0; oh < H_out; ++oh) {
                    for (std::size_t ow = 0; ow < W_out; ++ow) {
                        float sum = bias.at(oc, 0, 0, 0);
                        for (std::size_t ic = 0; ic < in_channels; ++ic) {
                            for (std::size_t kh = 0; kh < kernel_size; ++kh) {
                                for (std::size_t kw = 0; kw < kernel_size; ++kw) {
                                    std::size_t ih = oh * stride + kh - padding;
                                    std::size_t iw = ow * stride + kw - padding;
                                    if (ih < H && iw < W) {
                                        sum += input.at(b, ic, ih, iw) *
                                            kernel.at(oc, ic, kh, kw);
                                    }
                                }
                            }
                        }
                        output.at(b, oc, oh, ow) = sum;
                    }
                }
            }
        }
        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        std::size_t batch_size = gradient.batch_size();
        std::size_t H = input_cache.height();
        std::size_t W = input_cache.width();
        Matrix input_gradients(batch_size, in_channels, H, W);

        kernel.zero_gradients();
        bias.zero_gradients();
        input_gradients.zero_gradients();

        auto [H_out, W_out] = get_output_dims(H, W);

        for (std::size_t oc = 0; oc < out_channels; ++oc) {
            float bias_grad = 0.0f;
            for (std::size_t b = 0; b < batch_size; ++b) {
                for (std::size_t oh = 0; oh < H_out; ++oh) {
                    for (std::size_t ow = 0; ow < W_out; ++ow) {
                        bias_grad += gradient.at(b, oc, oh, ow);
                    }
                }
            }
            bias.add_gradient(oc, 0, 0, 0, bias_grad);
        }

        for (std::size_t b = 0; b < batch_size; ++b) {
            for (std::size_t oc = 0; oc < out_channels; ++oc) {
                for (std::size_t oh = 0; oh < H_out; ++oh) {
                    for (std::size_t ow = 0; ow < W_out; ++ow) {
                        float grad = gradient.at(b, oc, oh, ow);

                        for (std::size_t ic = 0; ic < in_channels; ++ic) {
                            for (std::size_t kh = 0; kh < kernel_size; ++kh) {
                                for (std::size_t kw = 0; kw < kernel_size; ++kw) {
                                    std::size_t ih = oh * stride + kh - padding;
                                    std::size_t iw = ow * stride + kw - padding;

                                    if (ih < H && iw < W) {
                                        float kernel_grad = input_cache.at(b, ic, ih, iw) * grad;
                                        kernel.add_gradient(oc, ic, kh, kw, kernel_grad);

                                        input_gradients.at(b, ic, ih, iw) +=
                                            kernel.at(oc, ic, kh, kw) * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return input_gradients;
    }

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