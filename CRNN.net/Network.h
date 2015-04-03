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

        void SetInput(FloatArray^ data);
        FloatArray^ Forward();
        System::String^ Translate(int k);
        array<int>^ InputDims();

    private:
        network *m_pnetwork;

    public:
        static void  TrainAndTestNetwork(System::String^ filename);
    };
}
