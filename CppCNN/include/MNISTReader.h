#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include "Matrix.h"

class MNISTReader {
private:
    static uint32_t reverseInt(uint32_t i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((uint32_t)ch1 << 24) + ((uint32_t)ch2 << 16) + ((uint32_t)ch3 << 8) + ch4;
    }

public:
    static Matrix readImageBatch(const std::string& path, size_t batchSize, size_t startIndex = 0) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file " + path);
        }

        uint32_t magic_number = 0;
        uint32_t number_of_images = 0;
        uint32_t rows = 0;
        uint32_t cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        file.read((char*)&rows, sizeof(rows));
        rows = reverseInt(rows);

        file.read((char*)&cols, sizeof(cols));
        cols = reverseInt(cols);

        if (startIndex + batchSize > number_of_images) {
            throw std::runtime_error("Batch size exceeds available images");
        }

        // grayscale, channel = 1
        Matrix batch(batchSize, 1, rows, cols);

        // Skip to the starting image
        file.seekg(16 + startIndex * rows * cols, std::ios::beg);

        // Read batch
        for (size_t i = 0; i < batchSize; i++) {
            for (size_t h = 0; h < rows; h++) {
                for (size_t w = 0; w < cols; w++) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    batch.at(i, 0, h, w) = temp / 255.0f;
                }
            }
        }

        return batch;
    }

    static Matrix readLabelBatch(const std::string& path, size_t batchSize, size_t startIndex = 0) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file " + path);
        }

        uint32_t magic_number = 0;
        uint32_t number_of_labels = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        if (startIndex + batchSize > number_of_labels) {
            throw std::runtime_error("Batch size exceeds available labels");
        }

        Matrix labels(batchSize, 1, 1, 1);

        file.seekg(8 + startIndex, std::ios::beg);

        // Read batch
        for (size_t i = 0; i < batchSize; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels.at(i, 0, 0, 0) = static_cast<float>(temp);
        }

        return labels;
    }
};