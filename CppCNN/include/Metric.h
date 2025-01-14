#pragma once
#include "Matrix.h"

class Metric {
public:
    static float computeAccuracy(const Matrix& logits, const Matrix& labels) {
        size_t correct = 0;
        size_t total = logits.batch_size();

        for (size_t i = 0; i < total; i++) {
            float maxLogit = logits.at(i, 0, 0, 0);
            size_t predictedClass = 0;

            for (size_t j = 1; j < logits.width(); j++) {
                if (logits.at(i, 0, 0, j) > maxLogit) {
                    maxLogit = logits.at(i, 0, 0, j);
                    predictedClass = j;
                }
            }

            if (static_cast<size_t>(labels.at(i, 0, 0, 0)) == predictedClass) {
                correct++;
            }
        }

        return static_cast<float>(correct) / total;
    }
};