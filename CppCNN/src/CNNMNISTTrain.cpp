#include <iostream>
#include <filesystem>
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
    // Conv2d(in_channel, out_channel, kernel_size, stride, padding)
    Sequential model = Sequential(Conv2d(1, 3, 3, 2, 0))
        .add(ReLU())
        .add(Conv2d(3, 6, 3, 2, 0))
        .add(ReLU())
        .add(Flatten())
        //.add(Dropout(0.5))
        .add(Linear(6 * 6 * 6, 10));

    const std::string train_images = std::string(DATA_DIR) + "/train-images.idx3-ubyte";
    const std::string train_labels = std::string(DATA_DIR) + "/train-labels.idx1-ubyte";
    const std::string test_images = std::string(DATA_DIR) + "/test-images.idx3-ubyte";
    const std::string test_labels = std::string(DATA_DIR) + "/test-labels.idx1-ubyte";
    const size_t batch_size = 32;
    const float learning_rate = 0.01f;
    const size_t total_epochs = 10;
    const size_t train_size = 60000;
    const size_t test_size = 10000;

    size_t start_epoch = 0;
    std::string last_model_path;
    size_t latest_epoch = 0;

    // Finding latest model
    for (const auto& entry : std::filesystem::directory_iterator(DATA_DIR)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("mnist_model_epoch_") != std::string::npos &&
            filename.find(".bin") != std::string::npos) {
            size_t pos = filename.find("epoch_");
            size_t dot_pos = filename.find(".bin");
            if (pos != std::string::npos && dot_pos != std::string::npos) {
                size_t epoch_num = std::stoi(filename.substr(pos + 6, dot_pos - (pos + 6)));
                if (epoch_num > latest_epoch) {
                    latest_epoch = epoch_num;
                    last_model_path = entry.path().string();
                }
            }
        }
    }

    try {
        if (!last_model_path.empty()) {
            std::ifstream file(last_model_path, std::ios::binary);
            if (file) {
                model.loadParameters(file);
                start_epoch = latest_epoch + 1;
                std::cout << "Latest model loaded successfully from " << last_model_path << std::endl;
                std::cout << "Continuing training from epoch " << start_epoch << std::endl;
            }
        }
        else {
            std::cout << "No previous model found. Starting training from epoch 0" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::cout << "Starting training from epoch 0" << std::endl;
    }

    // Training loop
    for (size_t epoch = start_epoch; epoch < start_epoch + total_epochs; epoch++) {
        float epoch_loss = 0.0f;
        size_t batches = 0;

        for (size_t start_idx = 0; start_idx < train_size; start_idx += batch_size) {
            size_t current_batch_size = std::min(batch_size, train_size - start_idx);

            Matrix images = MNISTReader::readImageBatch(train_images, current_batch_size, start_idx);
            Matrix labels = MNISTReader::readLabelBatch(train_labels, current_batch_size, start_idx);

            Matrix logits = model.forward(images);
            float loss = CrossEntropyLoss::compute(logits, labels);
            Matrix gradient = CrossEntropyLoss::compute_gradient(logits, labels);
            model.backward(gradient);
            model.update_parameters(learning_rate);

            epoch_loss += loss;
            batches++;

            if (batches % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Batch " << batches
                    << ", Loss: " << epoch_loss / batches << std::endl;
            }
        }

        float test_accuracy = evaluateModel(model, test_images, test_labels,
            test_size, batch_size);

        std::cout << "Epoch " << epoch << " completed. "
            << "Average loss: " << epoch_loss / batches
            << ", Test accuracy: " << test_accuracy * 100 << "%" << std::endl;

        // Save model
        std::string model_path = std::string(DATA_DIR) + "/mnist_model_epoch_" + std::to_string(epoch) + ".bin";
        std::ofstream out_file(model_path, std::ios::binary);
        if (!out_file) {
            throw std::runtime_error("Cannot create file: " + model_path);
        }
        model.saveParameters(out_file);
        std::cout << "Model saved to " << model_path << std::endl;
    }

    return 0;
}

