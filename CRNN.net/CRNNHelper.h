#pragma once
#include "../CRNN/common.h"
#include "../CRNN/memory.h"

using namespace System;

namespace CRNNnet {
    public ref class CRNNHelper
    {
    public:
        static std::string MarshalString(String ^ s);
    };
}
