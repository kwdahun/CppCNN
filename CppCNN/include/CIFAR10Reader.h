#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include "Matrix.h"

class CIFAR10Reader {
private:
    static constexpr size_t IMAGE_SIZE = 32;
    static constexpr size_t CHANNELS = 3;
    static constexpr size_t LABEL_SIZE = 1;
    static constexpr size_t PIXELS_PER_CHANNEL = IMAGE_SIZE * IMAGE_SIZE;
    static constexpr size_t BYTES_PER_IMAGE = CHANNELS * PIXELS_PER_CHANNEL;
    static constexpr size_t ROW_SIZE = LABEL_SIZE + BYTES_PER_IMAGE;
    static constexpr size_t IMAGES_PER_FILE = 10000;

public:
    static Matrix readBatch(const std::string& path, size_t batchSize, size_t startIndex = 0) {

        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File does not exist: " + path);
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path +
                " (Make sure the file exists and you have proper permissions)");
        }

        if (startIndex >= IMAGES_PER_FILE) {
            throw std::runtime_error("Start index " + std::to_string(startIndex) +
                " exceeds file size (max: " + std::to_string(IMAGES_PER_FILE) + ")");
        }

        batchSize = std::min(batchSize, IMAGES_PER_FILE - startIndex);

        Matrix batch(batchSize, CHANNELS, IMAGE_SIZE, IMAGE_SIZE);

        try {

            file.seekg(startIndex * ROW_SIZE, std::ios::beg);
            if (file.fail()) {
                throw std::runtime_error("Failed to seek to position " +
                    std::to_string(startIndex * ROW_SIZE));
            }

            for (size_t i = 0; i < batchSize; i++) {

                file.seekg(LABEL_SIZE, std::ios::cur);
                if (file.fail()) {
                    throw std::runtime_error("Failed to skip label at image " + std::to_string(i));
                }

                for (size_t c = 0; c < CHANNELS; c++) {
                    for (size_t h = 0; h < IMAGE_SIZE; h++) {
                        for (size_t w = 0; w < IMAGE_SIZE; w++) {
                            unsigned char pixel;
                            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                            if (file.fail()) {
                                throw std::runtime_error("Failed to read pixel at image " +
                                    std::to_string(i) + ", channel " + std::to_string(c) +
                                    ", position (" + std::to_string(h) + "," + std::to_string(w) + ")");
                            }
                            batch.at(i, c, h, w) = static_cast<float>(pixel) / 255.0f;
                        }
                    }
                }
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Error reading file " + path + ": " + e.what());
        }

        return batch;
    }

    static Matrix readLabels(const std::string& path, size_t batchSize, size_t startIndex = 0) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File does not exist: " + path);
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path +
                " (Make sure the file exists and you have proper permissions)");
        }

        if (startIndex >= IMAGES_PER_FILE) {
            throw std::runtime_error("Start index exceeds file size");
        }

        batchSize = std::min(batchSize, IMAGES_PER_FILE - startIndex);

        Matrix labels(batchSize, 1, 1, 1);

        try {

            file.seekg(startIndex * ROW_SIZE, std::ios::beg);

            for (size_t i = 0; i < batchSize; i++) {
                unsigned char label;
                file.read(reinterpret_cast<char*>(&label), sizeof(label));
                labels.at(i, 0, 0, 0) = static_cast<float>(label);

                file.seekg(BYTES_PER_IMAGE, std::ios::cur);
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Error reading labels from " + path + ": " + e.what());
        }

        return labels;
    }
};