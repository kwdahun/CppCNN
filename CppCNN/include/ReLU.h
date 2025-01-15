#pragma once
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"

class ReLU : public Layer {
private:
    Matrix input_cache;

public:
    ReLU() : input_cache(1, 1, 1, 1) {}

    Matrix forward(const Matrix& input) override {
        input_cache = input;

        Matrix output(input.batch_size(), input.channels(), input.height(), input.width());

#pragma omp parallel for collapse(4)
        for (size_t b = 0; b < input.batch_size(); ++b) {
            for (size_t c = 0; c < input.channels(); ++c) {
                for (size_t h = 0; h < input.height(); ++h) {
                    for (size_t w = 0; w < input.width(); ++w) {
                        output.at(b, c, h, w) = std::max(0.0f, input.at(b, c, h, w));
                    }
                }
            }
        }
        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        // ReLU derivative: 1 if input > 0, 0 otherwise
        Matrix output(gradient.batch_size(), gradient.channels(), gradient.height(), gradient.width());
        for (size_t b = 0; b < gradient.batch_size(); ++b) {
            for (size_t c = 0; c < gradient.channels(); ++c) {
                for (size_t h = 0; h < gradient.height(); ++h) {
                    for (size_t w = 0; w < gradient.width(); ++w) {
                        output.at(b, c, h, w) = input_cache.at(b, c, h, w) > 0 ?
                            gradient.at(b, c, h, w) : 0.0f;
                    }
                }
            }
        }
        return output;
    }

    void update_parameters(float learning_rate) override {
        // no parameters
    }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::ReLU);
    }

    void saveParameters(std::ofstream& file) const override {
        // No parameters
    }

    void loadParameters(std::ifstream& file) override {
        // No parameters
    }
};