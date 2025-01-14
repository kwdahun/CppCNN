#include <iostream>
#include <filesystem>
#include "Matrix.h"
#include "Layer.h"
#include "Linear.h"
#include "Conv2d.h"
#include "ReLU.h"
#include "Flatten.h"
#include "Dropout.h"
#include "MaxPool2d.h"
#include "Sequential.h"
#include "CrossEntropyLoss.h"
#include "CIFAR10Reader.h"
#include "Metric.h"

float evaluateModel(Sequential& model, const std::string& dataDir,
    size_t test_size, size_t batch_size) {
    float total_accuracy = 0.0f;
    size_t num_batches = 0;

    for (size_t start_idx = 0; start_idx < test_size; start_idx += batch_size) {
        size_t current_batch_size = std::min(batch_size, test_size - start_idx);

        Matrix images = CIFAR10Reader::readBatch(dataDir + "/test_batch.bin",
            current_batch_size, start_idx);
        Matrix labels = CIFAR10Reader::readLabels(dataDir + "/test_batch.bin",
            current_batch_size, start_idx);

        Matrix logits = model.forward(images);
        float accuracy = Metric::computeAccuracy(logits, labels);
        total_accuracy += accuracy;
        num_batches++;
    }

    return total_accuracy / num_batches;
}

int main() {
    const std::string data_dir = std::string(DATA_DIR);
    std::cout << "Checking data directory: " << data_dir << std::endl;

    if (!std::filesystem::exists(data_dir)) {
        std::cerr << "Error: Data directory does not exist: " << data_dir << std::endl;
        return 1;
    }

    Sequential model = Sequential(Conv2d(3, 6, 5, 1, 2))     // 32x32 -> 32x32
        .add(ReLU())
        .add(MaxPool2d(3, 2))                                 // 32x32 -> 15x15
        .add(Dropout(0.2))

        .add(Conv2d(6, 12, 5, 1, 2))                       // 15x15 -> 15x15
        .add(ReLU())
        .add(MaxPool2d(3, 2))                                // 15x15 -> 7x7
        .add(Dropout(0.3))

        .add(Conv2d(12, 15, 3, 1, 1))                      // 7x7 -> 7x7
        .add(ReLU())

        .add(Conv2d(15, 15, 3, 1, 1))                      // 7x7 -> 7x7
        .add(ReLU())

        .add(Conv2d(15, 3, 3, 1, 1))                      // 7x7 -> 7x7
        .add(ReLU())
        .add(MaxPool2d(3, 2))                                // 7x7 -> 3x3

        .add(Flatten())                                       // 128 * 3 * 3
        .add(Dropout(0.5))
        .add(Linear(3 * 3 * 3, 128))
        .add(ReLU())
        .add(Dropout(0.5))
        .add(Linear(128, 64))
        .add(ReLU())
        .add(Linear(64, 10));


    const size_t batch_size = 64;  // Increased from 32
    const float learning_rate = 0.001f;  // Decreased from 0.01
    const size_t total_epochs = 20;
    const size_t train_size = 50000;
    const size_t test_size = 10000;

    // Rest of the training code remains the same
    size_t start_epoch = 0;
    std::string last_model_path;
    int latest_epoch = -1;

    // Finding latest model
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("cifar10_model_epoch_") != std::string::npos &&
            filename.find(".bin") != std::string::npos) {
            size_t pos = filename.find("epoch_");
            size_t dot_pos = filename.find(".bin");
            if (pos != std::string::npos && dot_pos != std::string::npos) {
                int epoch_num = std::stoi(filename.substr(pos + 6, dot_pos - (pos + 6)));
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

        model.setTrainingMode(true);  // Explicitly set training mode

        // Iterate through all 5 training batch files
        for (size_t file_idx = 1; file_idx <= 5; file_idx++) {
            std::string batch_file = data_dir + "/data_batch_" + std::to_string(file_idx) + ".bin";

            for (size_t start_idx = 0; start_idx < 10000; start_idx += batch_size) {
                size_t current_batch_size = std::min(batch_size, 10000 - start_idx);

                Matrix images = CIFAR10Reader::readBatch(batch_file, current_batch_size, start_idx);
                Matrix labels = CIFAR10Reader::readLabels(batch_file, current_batch_size, start_idx);
                //images.print_color();
                //labels.print();

                Matrix logits = model.forward(images);
                float loss = CrossEntropyLoss::compute(logits, labels);
                Matrix gradient = CrossEntropyLoss::compute_gradient(logits, labels);
                model.backward(gradient);
                model.update_parameters(learning_rate);

                epoch_loss += loss;
                batches++;

                if (batches % 1 == 0) {
                    std::cout << "Epoch " << epoch << ", Batch " << batches
                        << ", Loss: " << epoch_loss / batches << std::endl;
                }
            }
        }

        // Evaluation mode
        model.setTrainingMode(false);
        float test_accuracy = evaluateModel(model, data_dir, test_size, batch_size);

        std::cout << "Epoch " << epoch << " completed. "
            << "Average loss: " << epoch_loss / batches
            << ", Test accuracy: " << test_accuracy * 100 << "%" << std::endl;

        // Save model
        std::string model_path = data_dir + "/cifar10_model_epoch_" + std::to_string(epoch) + ".bin";
        std::ofstream out_file(model_path, std::ios::binary);
        if (!out_file) {
            throw std::runtime_error("Cannot create file: " + model_path);
        }
        model.saveParameters(out_file);
        std::cout << "Model saved to " << model_path << std::endl;
    }

    return 0;
}