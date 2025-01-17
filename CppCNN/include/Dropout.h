#pragma once
#include <omp.h>
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"
#include <random>

class Dropout : public Layer {
private:
    float dropout_rate;
    Matrix mask;
    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution;

public:
    Dropout(float dropout_rate = 0.5f)
        : dropout_rate(dropout_rate)
        , mask(1, 1, 1, 1)
        , generator(std::random_device{}())
        , distribution(0.0f, 1.0f)
    {
        if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
            throw std::invalid_argument("Dropout rate must be in range [0, 1)");
        }
    }

    Matrix forward(const Matrix& input) override {
        if (!is_training) {
            return input * (1.0f - dropout_rate);
        }

        mask = Matrix(input.batch_size(), input.channels(), input.height(), input.width());
        Matrix output(input.batch_size(), input.channels(), input.height(), input.width());
        float scale = 1.0f / (1.0f - dropout_rate);

#pragma omp parallel
        {
            // Create thread-local random number generator
            std::mt19937 local_generator(generator());
            std::uniform_real_distribution<float> local_distribution(0.0f, 1.0f);

#pragma omp for collapse(3)
            for (size_t b = 0; b < input.batch_size(); ++b) {
                for (size_t c = 0; c < input.channels(); ++c) {
                    for (size_t h = 0; h < input.height(); ++h) {
#pragma omp simd
                        for (size_t w = 0; w < input.width(); ++w) {
                            float rand_val = local_distribution(local_generator);
                            float mask_val = rand_val > dropout_rate ? 1.0f : 0.0f;
                            mask.at(b, c, h, w) = mask_val;
                            output.at(b, c, h, w) = input.at(b, c, h, w) * mask_val * scale;
                        }
                    }
                }
            }
        }

        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        if (!is_training) {
            return gradient * (1.0f - dropout_rate);
        }

        Matrix output(gradient.batch_size(), gradient.channels(), gradient.height(), gradient.width());
        float scale = 1.0f / (1.0f - dropout_rate);

#pragma omp parallel for collapse(3)
        for (size_t b = 0; b < gradient.batch_size(); ++b) {
            for (size_t c = 0; c < gradient.channels(); ++c) {
                for (size_t h = 0; h < gradient.height(); ++h) {
#pragma omp simd
                    for (size_t w = 0; w < gradient.width(); ++w) {
                        output.at(b, c, h, w) = gradient.at(b, c, h, w) * mask.at(b, c, h, w) * scale;
                    }
                }
            }
        }

        return output;
    }

    void update_parameters(float learning_rate) override {
        // No parameters to update
    }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::Dropout);
    }

    void saveParameters(std::ofstream& file) const override {
        file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(dropout_rate));
    }

    void loadParameters(std::ifstream& file) override {
        file.read(reinterpret_cast<char*>(&dropout_rate), sizeof(dropout_rate));
    }

    float getDropoutRate() const { return dropout_rate; }

    void setDropoutRate(float rate) {
        if (rate < 0.0f || rate >= 1.0f) {
            throw std::invalid_argument("Dropout rate must be in range [0, 1)");
        }
        dropout_rate = rate;
    }
};