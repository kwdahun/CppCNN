#pragma once
#include <cmath>
#include "Matrix.h"

class CrossEntropyLoss {
public:
    static void softmax(Matrix& input, size_t batch_idx) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t w = 0; w < input.width(); w++) {
            max_val = std::max(max_val, input.at(batch_idx, 0, 0, w));
        }

        float sum = 0.0f;
        for (size_t w = 0; w < input.width(); w++) {
            float val = std::exp(input.at(batch_idx, 0, 0, w) - max_val);
            input.at(batch_idx, 0, 0, w) = val;
            sum += val;
        }

        for (size_t w = 0; w < input.width(); w++) {
            input.at(batch_idx, 0, 0, w) /= sum;
        }
    }

    static Matrix apply_softmax(const Matrix& input) {
        Matrix output(input);
        for (size_t b = 0; b < output.batch_size(); b++) {
            softmax(output, b);
        }
        return output;
    }

    static float compute(const Matrix& logits, const Matrix& targets) {
        Matrix predictions = apply_softmax(logits);
        float loss = 0.0f;
        size_t batch_size = predictions.batch_size();

        for (size_t b = 0; b < batch_size; b++) {
            int target = static_cast<int>(targets.at(b, 0, 0, 0));
            float pred = predictions.at(b, 0, 0, target);
            loss -= std::log(std::max(pred, 1e-7f));
        }

        return loss / batch_size;
    }

    static Matrix compute_gradient(const Matrix& logits, const Matrix& targets) {
        Matrix predictions = apply_softmax(logits);
        Matrix gradient(predictions.batch_size(), 1, 1, predictions.width());
        size_t batch_size = predictions.batch_size();

        for (size_t b = 0; b < batch_size; b++) {
            int target = static_cast<int>(targets.at(b, 0, 0, 0));

            // Gradient for L about y_hat is y - y_hat
            for (size_t w = 0; w < predictions.width(); w++) {
                gradient.at(b, 0, 0, w) = predictions.at(b, 0, 0, w);
            }
            gradient.at(b, 0, 0, target) -= 1.0f;
        }

        gradient = gradient * (1.0f / batch_size);

        return gradient;
    }
};