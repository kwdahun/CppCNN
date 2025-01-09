#pragma once
#include <stdint.h>

enum class LayerType : uint8_t {
    Sequential = 0,
    Conv2d = 1,
    Linear = 2,
    ReLU = 3,
    Flatten = 4
};