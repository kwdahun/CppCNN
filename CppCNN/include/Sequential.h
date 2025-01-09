#pragma once
#include <vector>
#include <memory>
#include <utility>
#include "Matrix.h"
#include "Layer.h"
#include "LayerType.h"
#include "Linear.h"

class Sequential : public Layer {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    Sequential() = default;
    Sequential(const Sequential&) = delete;
    Sequential(Sequential&& other) noexcept
        : layers(std::move(other.layers)) {}

    Sequential& operator=(Sequential&& other) noexcept {
        if (this != &other) {
            layers = std::move(other.layers);
        }
        return *this;
    }

    template<typename T>
    Sequential(T&& layer) {
        layers.push_back(std::make_unique<std::decay_t<T>>(std::forward<T>(layer)));
    }

    template<typename T>
    Sequential add(T&& layer)&& {
        layers.push_back(std::make_unique<std::decay_t<T>>(std::forward<T>(layer)));
        return std::move(*this);
    }

    Matrix forward(const Matrix& input) override {
        Matrix output = input;
        for (const auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    Matrix backward(const Matrix& gradient) override {
        Matrix grad = gradient;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    void update_parameters(float learning_rate) override {
        for (auto& layer : layers) {
            layer->update_parameters(learning_rate);
        }
    }

    uint8_t getLayerType() const override {
        return static_cast<uint8_t>(LayerType::Sequential);
    }

    void saveParameters(std::ofstream& file) const override {
        size_t num_layers = layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

        for (const auto& layer : layers) {
            uint8_t layer_type = layer->getLayerType();
            file.write(reinterpret_cast<const char*>(&layer_type), sizeof(layer_type));

            layer->saveParameters(file);
        }
    }

    void loadParameters(std::ifstream& file) override {
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        if (num_layers != layers.size()) {
            throw std::runtime_error("Model architecture mismatch: different number of layers");
        }

        for (size_t i = 0; i < num_layers; ++i) {
            uint8_t loaded_type;
            file.read(reinterpret_cast<char*>(&loaded_type), sizeof(loaded_type));

            if (loaded_type != layers[i]->getLayerType()) {
                throw std::runtime_error("Model architecture mismatch: different layer types");
            }

            layers[i]->loadParameters(file);
        }
    }
};