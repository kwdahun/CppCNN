set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)

file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(MNIST_TRAIN_MAIN "src/CNNMNISTTrain.cpp")
set(MNIST_INFERENCE_MAIN "src/CNNMNISTInference.cpp")
set(CIFAR10_TRAIN_MAIN "src/CNNCIFAR10Train.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(mnist_train 
    ${MNIST_TRAIN_MAIN}
    ${HEADERS}
)
add_executable(mnist_inference
    ${MNIST_INFERENCE_MAIN}
    ${HEADERS}
)
add_executable(cifar10_train
    ${CIFAR10_TRAIN_MAIN}
    ${HEADERS}
)

target_compile_definitions(mnist_train PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/mnist"
)
target_compile_definitions(mnist_inference PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/mnist"
)
target_compile_definitions(cifar10_train PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/cifar10"
)