#pragma once
#include <cmath>
#include <omp.h>
#include <cblas.h>
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

    Matrix m_weight;
    Matrix v_weight;
    Matrix m_bias;
    Matrix v_bias;
    size_t t;
    float beta1;
    float beta2;
    float epsilon;

public:
    Linear(std::size_t in_features, std::size_t out_features)
        : in_features(in_features), out_features(out_features),
        weight(1, 1, out_features, in_features),
        bias(1, 1, 1, out_features),
        input_cache(1, 1, 1, 1),

        m_weight(1, 1, out_features, in_features),
        v_weight(1, 1, out_features, in_features),
        m_bias(1, 1, 1, out_features),
        v_bias(1, 1, 1, out_features),
        t(0),
        beta1(0.9f),
        beta2(0.999f),
        epsilon(1e-8f)
    {
        float limit = std::sqrt(2.0f / in_features);
        for (std::size_t i = 0; i < out_features; i++) {
            for (std::size_t j = 0; j < in_features; j++) {
                float random_val = ((float)rand() / RAND_MAX) * 2 * limit - limit;
                weight.at(0, 0, i, j) = random_val;
                m_weight.at(0, 0, i, j) = 0.0f;
                v_weight.at(0, 0, i, j) = 0.0f;
            }
            bias.at(0, 0, 0, i) = 0.0f;
            m_bias.at(0, 0, 0, i) = 0.0f;
            v_bias.at(0, 0, 0, i) = 0.0f;
        }
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        std::size_t batch_size = input.batch_size();
        Matrix output(batch_size, 1, 1, out_features);

        int M = batch_size;
        int N = out_features;
        int K = in_features;

        cblas_sgemm(CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M,
            N,
            K,
            1.0f,
            &input_cache.at(0, 0, 0, 0),
            K,
            &weight.at(0, 0, 0, 0),
            K,
            0.0f,
            &output.at(0, 0, 0, 0),
            N);

#pragma omp parallel for
        for (std::size_t b = 0; b < batch_size; b++) {
#pragma omp simd
            for (std::size_t i = 0; i < out_features; i++) {
                output.at(b, 0, 0, i) += bias.at(0, 0, 0, i);
            }
        }

        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        std::size_t batch_size = gradient.batch_size();
        Matrix weight_gradients(weight.batch_size(), weight.channels(), weight.height(), weight.width());
        Matrix bias_gradients(bias.batch_size(), bias.channels(), bias.height(), bias.width());
        Matrix input_gradients(input_cache.batch_size(), input_cache.channels(),
            input_cache.height(), input_cache.width());

#pragma omp parallel for
        for (size_t i = 0; i < out_features; ++i) {
            float sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                sum += gradient.at(b, 0, 0, i);
            }
            bias.add_gradient(0, 0, 0, i, sum);
        }

        int M = out_features;
        int N = in_features;
        int K = batch_size;

        cblas_sgemm(CblasRowMajor,
            CblasTrans,
            CblasNoTrans,
            M,
            N,
            K,
            1.0f,
            &gradient.at(0, 0, 0, 0),
            M,
            &input_cache.at(0, 0, 0, 0),
            N,
            0.0f,
            &weight_gradients.at(0, 0, 0, 0),
            N);

        M = batch_size;
        N = in_features;
        K = out_features;

        cblas_sgemm(CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            1.0f,
            &gradient.at(0, 0, 0, 0),
            K,
            &weight.at(0, 0, 0, 0),
            N,
            0.0f,
            &input_gradients.at(0, 0, 0, 0),
            N);

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                weight.add_gradient(0, 0, i, j, weight_gradients.at(0, 0, i, j));
            }
        }

        return input_gradients;
    }

    void update_parameters(float learning_rate) override {
        t++;

        float correction1 = 1.0f / (1.0f - std::pow(beta1, t));
        float correction2 = 1.0f / (1.0f - std::pow(beta2, t));

        for (size_t i = 0; i < out_features; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                float grad = weight.get_gradient(0, 0, i, j);

                m_weight.at(0, 0, i, j) = beta1 * m_weight.at(0, 0, i, j) + (1 - beta1) * grad;

                v_weight.at(0, 0, i, j) = beta2 * v_weight.at(0, 0, i, j) + (1 - beta2) * grad * grad;

                float m_hat = m_weight.at(0, 0, i, j) * correction1;
                float v_hat = v_weight.at(0, 0, i, j) * correction2;

                weight.at(0, 0, i, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }

            float bias_grad = bias.get_gradient(0, 0, 0, i);

            m_bias.at(0, 0, 0, i) = beta1 * m_bias.at(0, 0, 0, i) + (1 - beta1) * bias_grad;
            v_bias.at(0, 0, 0, i) = beta2 * v_bias.at(0, 0, 0, i) + (1 - beta2) * bias_grad * bias_grad;

            float m_hat_bias = m_bias.at(0, 0, 0, i) * correction1;
            float v_hat_bias = v_bias.at(0, 0, 0, i) * correction2;

            bias.at(0, 0, 0, i) -= learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);
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