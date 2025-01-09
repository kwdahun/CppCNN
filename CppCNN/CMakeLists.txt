set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)

file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(TRAIN_MAIN "src/CNNMNISTTrain.cpp")
set(INFERENCE_MAIN "src/CNNMNISTInference.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(mnist_train 
    ${TRAIN_MAIN}
    ${HEADERS}
)
add_executable(mnist_inference
    ${INFERENCE_MAIN}
    ${HEADERS}
)

target_compile_definitions(mnist_train PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
target_compile_definitions(mnist_inference PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)