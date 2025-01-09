#pragma once
#include <cmath>
#include "Layer.h"
#include "LayerType.h"
#include "Matrix.h"


class Linear : public Layer {
private:
    Matrix weight;
    Matrix bias;
    Matrix input_cache;
    std::size_t in_features;
    std::size_t out_features;
public:
    Linear(std::size_t in_features, std::size_t out_features)
        : in_features(in_features), out_features(out_features),
        weight(1, 1, out_features, in_features),
        bias(1, 1, 1, out_features),
        input_cache(1, 1, 1, 1)
    {
        float limit = std::sqrt(1.0f / in_features);
        for (std::size_t i = 0; i < out_features; i++) {
            for (std::size_t j = 0; j < in_features; j++) {
                float random_val = ((float)rand() / RAND_MAX) * 2 * limit - limit;
                weight.at(0, 0, i, j) = random_val;
            }
            bias.at(0, 0, 0, i) = 0.0f;
        }
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input;

        std::size_t batch_size = input.batch_size();
        Matrix output(batch_size, 1, 1, out_features);
        for (std::size_t b = 0; b < batch_size; b++) {
            for (std::size_t i = 0; i < out_features; i++) {
                float sum = 0.0f;
                for (std::size_t j = 0; j < in_features; j++) {
                    sum += input.at(b, 0, 0, j) * weight.at(0, 0, i, j);
                }
                output.at(b, 0, 0, i) = sum + bias.at(0, 0, 0, i);
            }
        }
        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        Matrix weight_gradients(weight.batch_size(), weight.channels(), weight.height(), weight.width());
        Matrix bias_gradients(bias.batch_size(), bias.channels(), bias.height(), bias.width());
        Matrix input_gradients(input_cache.batch_size(), input_cache.channels(),
            input_cache.height(), input_cache.width());

        weight_gradients.zero_gradients();
        bias_gradients.zero_gradients();
        input_gradients.zero_gradients();

        for (size_t b = 0; b < gradient.batch_size(); ++b) {
            for (size_t i = 0; i < out_features; ++i) {
                for (size_t j = 0; j < in_features; ++j) {
                    float grad = input_cache.at(b, 0, 0, j) * gradient.at(b, 0, 0, i);
                    weight_gradients.at(0, 0, i, j) += grad;

                    float input_grad = weight.at(0, 0, i, j) * gradient.at(b, 0, 0, i);
                    input_gradients.at(b, 0, 0, j) += input_grad;
                }
                bias_gradients.at(0, 0, 0, i) += gradient.at(b, 0, 0, i);
            }
        }

        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                weight.add_gradient(0, 0, i, j, weight_gradients.at(0, 0, i, j));
            }
            bias.add_gradient(0, 0, 0, i, bias_gradients.at(0, 0, 0, i));
        }

        return input_gradients;
    }

    void update_parameters(float learning_rate) override {
        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                float grad = weight.get_gradient(0, 0, i, j);
                weight.at(0, 0, i, j) -= learning_rate * grad;
            }
            float bias_grad = bias.get_gradient(0, 0, 0, i);
            bias.at(0, 0, 0, i) -= learning_rate * bias_grad;
        }
        weight.zero_gradients();
        bias.zero_gradients();
    }

    std::size_t get_in_features() const { return in_features; }
    std::size_t get_out_features() const { return out_features; }
    const Matrix& get_weight() const { return weight; }
    const Matrix& get_bias() const { return bias; }
    void set_weight(const Matrix& w) { weight = w; }
    void set_bias(const Matrix& b) { bias = b; }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::Linear);
    }

    void saveParameters(std::ofstream& file) const override {
        file.write(reinterpret_cast<const char*>(&in_features), sizeof(in_features));
        file.write(reinterpret_cast<const char*>(&out_features), sizeof(out_features));

        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                float w = weight.at(0, 0, i, j);
                file.write(reinterpret_cast<const char*>(&w), sizeof(float));
            }
        }

        for (size_t i = 0; i < out_features; ++i) {
            float b = bias.at(0, 0, 0, i);
            file.write(reinterpret_cast<const char*>(&b), sizeof(float));
        }
    }

    void loadParameters(std::ifstream& file) override {
        file.read(reinterpret_cast<char*>(&in_features), sizeof(in_features));
        file.read(reinterpret_cast<char*>(&out_features), sizeof(out_features));

        weight = Matrix(1, 1, out_features, in_features);
        bias = Matrix(1, 1, 1, out_features);

        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                float w;
                file.read(reinterpret_cast<char*>(&w), sizeof(float));
                weight.at(0, 0, i, j) = w;
            }
        }

        for (size_t i = 0; i < out_features; ++i) {
            float b;
            file.read(reinterpret_cast<char*>(&b), sizeof(float));
            bias.at(0, 0, 0, i) = b;
        }
    }
};