#pragma once
#include <omp.h>
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"

class Flatten : public Layer {
private:
    size_t original_channels;
    size_t original_height;
    size_t original_width;

public:
    Flatten() : original_channels(0), original_height(0), original_width(0) {};

    Matrix forward(const Matrix& input) override {
        original_channels = input.channels();
        original_height = input.height();
        original_width = input.width();

        size_t batch_size = input.batch_size();
        size_t flattened_size = original_channels * original_height * original_width;
        Matrix output(batch_size, 1, 1, flattened_size);

#pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < original_channels; ++c) {
                size_t base_idx = c * original_height * original_width;
                for (size_t h = 0; h < original_height; ++h) {
#pragma omp simd
                    for (size_t w = 0; w < original_width; ++w) {
                        size_t idx = base_idx + h * original_width + w;
                        output.at(b, 0, 0, idx) = input.at(b, c, h, w);
                    }
                }
            }
        }
        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        size_t batch_size = gradient.batch_size();
        Matrix output(batch_size, original_channels, original_height, original_width);

#pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < original_channels; ++c) {
                size_t base_idx = c * original_height * original_width;
                for (size_t h = 0; h < original_height; ++h) {
#pragma omp simd
                    for (size_t w = 0; w < original_width; ++w) {
                        size_t idx = base_idx + h * original_width + w;
                        output.at(b, c, h, w) = gradient.at(b, 0, 0, idx);
                    }
                }
            }
        }
        return output;
    }

    void update_parameters(float learning_rate) override {
        // no parameter update
    }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::Flatten);
    }

    void saveParameters(std::ofstream& file) const override {
        file.write(reinterpret_cast<const char*>(&original_channels), sizeof(original_channels));
        file.write(reinterpret_cast<const char*>(&original_height), sizeof(original_height));
        file.write(reinterpret_cast<const char*>(&original_width), sizeof(original_width));
    }

    void loadParameters(std::ifstream& file) override {
        file.read(reinterpret_cast<char*>(&original_channels), sizeof(original_channels));
        file.read(reinterpret_cast<char*>(&original_height), sizeof(original_height));
        file.read(reinterpret_cast<char*>(&original_width), sizeof(original_width));
    }
};