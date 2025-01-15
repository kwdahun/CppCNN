#pragma once
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"

class MaxPool2d : public Layer {
private:
    Matrix indices_cache;  // Store indices for backprop
    size_t kernel_size;
    size_t stride;

    size_t last_input_height;
    size_t last_input_width;

public:
    MaxPool2d(size_t kernel_size, size_t stride)
        : kernel_size(kernel_size)
        , stride(stride)
        , indices_cache(1, 1, 1, 1)
        , last_input_height(0)
        , last_input_width(0) {}

    Matrix forward(const Matrix& input) override {
        last_input_height = input.height();
        last_input_width = input.width();

        size_t batch_size = input.batch_size();
        size_t channels = input.channels();
        size_t height = input.height();
        size_t width = input.width();

        size_t output_height = (height - kernel_size) / stride + 1;
        size_t output_width = (width - kernel_size) / stride + 1;

        Matrix output(batch_size, channels, output_height, output_width);
        // Store h, w coordinates side by side in width dimension
        indices_cache = Matrix(batch_size, channels, output_height, output_width * 2);

#pragma omp parallel for collapse(4)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < output_height; ++h) {
                    for (size_t w = 0; w < output_width; ++w) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        size_t max_h = 0, max_w = 0;

                        // Find maximum in the kernel window
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                size_t ih = h * stride + kh;
                                size_t iw = w * stride + kw;

                                float val = input.at(b, c, ih, iw);
                                if (val > max_val) {
                                    max_val = val;
                                    max_h = ih;
                                    max_w = iw;
                                }
                            }
                        }

                        output.at(b, c, h, w) = max_val;
                        // Store indices side by side in width dimension
                        indices_cache.at(b, c, h, w * 2) = static_cast<float>(max_h);
                        indices_cache.at(b, c, h, w * 2 + 1) = static_cast<float>(max_w);
                    }
                }
            }
        }

        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        size_t batch_size = gradient.batch_size();
        size_t channels = gradient.channels();

        Matrix input_gradients(batch_size, channels, last_input_height, last_input_width);
        input_gradients.zero_gradients();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < gradient.height(); ++h) {
                    for (size_t w = 0; w < gradient.width(); ++w) {
                        size_t max_h = static_cast<size_t>(indices_cache.at(b, c, h, w * 2));
                        size_t max_w = static_cast<size_t>(indices_cache.at(b, c, h, w * 2 + 1));
                        input_gradients.at(b, c, max_h, max_w) += gradient.at(b, c, h, w);
                    }
                }
            }
        }
        return input_gradients;
    }

    void update_parameters(float learning_rate) override {
        // MaxPool layer has no learnable parameters
    }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::MaxPool2d);
    }

    void saveParameters(std::ofstream& file) const override {
        file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
        file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    }

    void loadParameters(std::ifstream& file) override {
        file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
        file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    }
};