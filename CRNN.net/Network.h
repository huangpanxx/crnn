#pragma once
#include "../CRNN/common.h"
#include "../CRNN/network.h"
#include "FloatArray.h"

namespace CRNNnet {
    public ref class Network
    {

    public:
        Network(System::String^ json, System::String^ plan);
        ~Network();

        void set_input(const FloatArray^ data);
        FloatArray^ forward();
        System::String^ translate(int k);
        array<int>^ input_dims();

    private:
        network *m_pnetwork;

    public:
        static void  TrainAndTestNetwork(System::String^ filename);
    };
}
