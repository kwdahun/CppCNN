# CNN Implementation in C++

This project implements a Convolutional Neural Network (CNN) from scratch in C++ for MNIST digit classification.

## Build Instructions

At project root directory:

```bash
# Create build directory
mkdir -p out/build

# Move to build directory
cd out/build

# Generate build files
cmake ../..

# Build the project
cmake --build .
```

## Running the Project

After building, while in the `out/build` directory:

### Training
```bash
./CppCNN/mnist_train
```

### Inference
```bash
./CppCNN/mnist_inference
```

## Project Structure
```
.
├── CMakeLists.txt
├── CMakePresets.json
├── CppCNN/
│   ├── CMakeLists.txt
│   ├── data/               # MNIST dataset and model files
│   ├── include/            # Header files
│   └── src/               # Source files
├── LICENSE
└── README.md
```

## Prerequisites

- C++20 compatible compiler
- CMake 3.0 or higher

## Data

The project expects MNIST dataset files in the `CppCNN/data` directory:
- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- test-images.idx3-ubyte
- test-labels.idx1-ubyte

After training, model files will be saved in the same directory with names like `mnist_model_epoch_N.bin` where N is the epoch number.
