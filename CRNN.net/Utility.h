#pragma once
#include "../CRNN/common.h"
#include "../CRNN/memory.h"

using namespace System;

namespace CRNNnet {
    public ref class Utility
    {
    public:
        static arraykd CreateArray(int size, array<float>^ data);
        static array2d CreateArray(int rows, int cols, array<float>^ data);
        static array3d CreateArray(int channel, int rows, int cols, array<float>^ data);
    };
}
