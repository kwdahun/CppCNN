#include <iostream>
#include "Matrix.h"
#include "Layer.h"
#include "Linear.h"
#include "Conv2d.h"
#include "ReLU.h"
#include "Flatten.h"
#include "Dropout.h"
#include "Sequential.h"
#include "CrossEntropyLoss.h"
#include "MNISTReader.h"


float computeAccuracy(const Matrix& logits, const Matrix& labels) {
    size_t correct = 0;
    size_t total = logits.batch_size();

    for (size_t i = 0; i < total; i++) {
        float maxLogit = logits.at(i, 0, 0, 0);
        size_t predictedClass = 0;

        for (size_t j = 1; j < 10; j++) {
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

float evaluateModel(Sequential& model, const std::string& test_images,
    const std::string& test_labels, size_t test_size,
    size_t batch_size) {
    float total_accuracy = 0.0f;
    size_t num_batches = 0;

    for (size_t start_idx = 0; start_idx < test_size; start_idx += batch_size) {
        size_t current_batch_size = std::min(batch_size, test_size - start_idx);

        Matrix images = MNISTReader::readImageBatch(test_images, current_batch_size, start_idx);
        Matrix labels = MNISTReader::readLabelBatch(test_labels, current_batch_size, start_idx);

        Matrix logits = model.forward(images);

        float accuracy = computeAccuracy(logits, labels);
        total_accuracy += accuracy;
        num_batches++;
    }

    return total_accuracy / num_batches;
}

int main() {
    Sequential model = Sequential(Conv2d(1, 3, 3, 2, 0))
        .add(ReLU())
        .add(Conv2d(3, 6, 3, 2, 0))
        .add(ReLU())
        .add(Flatten())
        .add(Dropout(0.5))
        .add(Linear(6 * 6 * 6, 10));

    std::string model_path = std::string(DATA_DIR) + "/mnist_model_epoch_19.bin";
    const std::string test_images = std::string(DATA_DIR) + "/test-images.idx3-ubyte";
    const std::string test_labels = std::string(DATA_DIR) + "/test-labels.idx1-ubyte";
    try {
        std::ifstream file(model_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open model file: " + model_path);
        }
        model.loadParameters(file);
        std::cout << "Model loaded successfully from " << model_path << std::endl;

        // Batch Test
        const size_t test_size = 10000;
        const size_t batch_size = 32;

        float test_accuracy = evaluateModel(model, test_images, test_labels,
            test_size, batch_size);
        std::cout << "Loaded model test accuracy: " << test_accuracy * 100 << "%" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    try {
        Matrix single_image = MNISTReader::readImageBatch(test_images, 1, 1111);
        Matrix logits = model.forward(single_image);

        size_t predicted_class = 0;
        float max_logit = logits.at(0, 0, 0, 0);
        for (size_t i = 1; i < 10; ++i) {
            if (logits.at(0, 0, 0, i) > max_logit) {
                max_logit = logits.at(0, 0, 0, i);
                predicted_class = i;
            }
        }
        single_image.print_color();
        std::cout << "Predicted digit: " << predicted_class << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}