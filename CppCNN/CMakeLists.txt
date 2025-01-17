set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenBLAS_DIR "C:/OpenBLAS")

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
include_directories(${OpenBLAS_DIR}/include)
link_directories(${OpenBLAS_DIR}/lib)

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

# Add custom command for each executable
add_custom_command(TARGET mnist_train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenBLAS_DIR}/bin/libopenblas.dll"
    $<TARGET_FILE_DIR:mnist_train>
)

add_custom_command(TARGET mnist_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenBLAS_DIR}/bin/libopenblas.dll"
    $<TARGET_FILE_DIR:mnist_inference>
)

add_custom_command(TARGET cifar10_train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenBLAS_DIR}/bin/libopenblas.dll"
    $<TARGET_FILE_DIR:cifar10_train>
)

target_link_libraries(mnist_train libopenblas)
target_link_libraries(mnist_inference libopenblas)
target_link_libraries(cifar10_train libopenblas)