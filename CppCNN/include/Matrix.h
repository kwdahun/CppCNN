#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>

class Matrix {
private:
    std::vector<std::vector<std::vector<std::vector<float>>>> data;
    std::vector<std::vector<std::vector<std::vector<float>>>> gradients;
    size_t batch_size_;
    size_t channels_;
    size_t height_;
    size_t width_;

public:
    // Constructor
    Matrix(size_t batch_size, size_t channels, size_t height, size_t width)
        : batch_size_(batch_size), channels_(channels), height_(height), width_(width) {
        data.resize(batch_size,
            std::vector<std::vector<std::vector<float>>>(channels,
                std::vector<std::vector<float>>(height,
                    std::vector<float>(width, 0.0f))));
        gradients.resize(batch_size,
            std::vector<std::vector<std::vector<float>>>(channels,
                std::vector<std::vector<float>>(height,
                    std::vector<float>(width, 0.0f))));
    }

    Matrix(size_t channels, size_t height, size_t width)
        : Matrix(1, channels, height, width) {}

    Matrix() : batch_size_(0), channels_(0), height_(0), width_(0) {}

    // Indexing
    float& at(size_t b, size_t c, size_t h, size_t w) {
        return data[b][c][h][w];
    }

    const float& at(size_t b, size_t c, size_t h, size_t w) const {
        return data[b][c][h][w];
    }

    // Operator Overloading
    Matrix operator*(const float& scalar) const {
        Matrix result(batch_size_, channels_, height_, width_);

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t c = 0; c < channels_; ++c) {
                for (size_t h = 0; h < height_; ++h) {
                    for (size_t w = 0; w < width_; ++w) {
                        result.at(b, c, h, w) = at(b, c, h, w) * scalar;
                    }
                }
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (width_ != other.height_) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }

        Matrix result(batch_size_, channels_, height_, other.width_);

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t c = 0; c < channels_; ++c) {
                for (size_t h = 0; h < height_; ++h) {
                    for (size_t w = 0; w < other.width_; ++w) {
                        float sum = 0.0f;
                        for (size_t k = 0; k < width_; ++k) {
                            sum += at(b, c, h, k) * other.at(b, c, k, w);
                        }
                        result.at(b, c, h, w) = sum;
                    }
                }
            }
        }

        return result;
    }

    Matrix operator+(const Matrix& other) const {
        check_dimensions(other);
        Matrix result(batch_size_, channels_, height_, width_);

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t c = 0; c < channels_; ++c) {
                for (size_t h = 0; h < height_; ++h) {
                    for (size_t w = 0; w < width_; ++w) {
                        result.at(b, c, h, w) = at(b, c, h, w) + other.at(b, c, h, w);
                    }
                }
            }
        }
        return result;
    }

    void check_dimensions(const Matrix& other) const {
        if (batch_size_ != other.batch_size_ ||
            channels_ != other.channels_ ||
            height_ != other.height_ ||
            width_ != other.width_) {
            throw std::invalid_argument("Matrix dimensions don't match");
        }
    }

    // Gradient
    void zero_gradients() {
        gradients = std::vector<std::vector<std::vector<std::vector<float>>>>(
            batch_size_,
            std::vector<std::vector<std::vector<float>>>(
                channels_,
                std::vector<std::vector<float>>(
                    height_,
                    std::vector<float>(width_, 0.0f))));
    }

    void add_gradient(size_t b, size_t c, size_t h, size_t w, float grad) {
        gradients[b][c][h][w] += grad;
    }

    float get_gradient(size_t b, size_t c, size_t h, size_t w) const {
        return gradients[b][c][h][w];
    }

    // Debugging
    void print() const {
        for (size_t b = 0; b < batch_size_; ++b) {
            std::cout << "Batch " << b << ":\n";
            for (size_t c = 0; c < channels_; ++c) {
                std::cout << " Channel " << c << ":\n";
                for (size_t h = 0; h < height_; ++h) {
                    std::cout << "  ";
                    for (size_t w = 0; w < width_; ++w) {
                        printf("%6.2f ", data[b][c][h][w]);
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    void print_color() const {
        for (size_t b = 0; b < batch_size_; ++b) {
            std::cout << "Batch " << b << ":\n";
            for (size_t c = 0; c < channels_; ++c) {
                std::cout << " Channel " << c << ":\n";
                for (size_t h = 0; h < height_; ++h) {
                    std::cout << "  ";
                    for (size_t w = 0; w < width_; ++w) {
                        // Convert float value [0,1] to grayscale intensity
                        int intensity = static_cast<int>(data[b][c][h][w] * 255);
                        // Print colored block using ANSI escape codes
                        std::cout << "\033[48;2;" << intensity << ";"
                            << intensity << ";" << intensity << "m  \033[0m";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

    // Getter
    size_t batch_size() const { return batch_size_; }
    size_t channels() const { return channels_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }
};