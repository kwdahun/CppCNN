#pragma once
#include <fstream>
#include "Matrix.h"

class Layer {
protected:
    bool is_training;

public:
    Layer() : is_training(true) {}
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient) = 0;
    virtual void update_parameters(float learning_rate) = 0;
    virtual ~Layer() = default;

    virtual uint8_t getLayerType() const = 0;
    virtual void saveParameters(std::ofstream& file) const = 0;
    virtual void loadParameters(std::ifstream& file) = 0;

    virtual void setTrainingMode(bool training) {
        is_training = training;
    }
    virtual bool isTraining() const {
        return is_training;
    }
};