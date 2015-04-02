#pragma once
#include "../CRNN/common.h"
#include "../CRNN/network.h"

using namespace System;

namespace CRNNnet {
    public ref class Network
    {

    public:
        Network(String^ json, String^ plan);
        ~Network();

        void set_input(const arraykd& data);
        arraykd forward();
        String^ translate(int k);
        array<int>^ input_dims();

    private:
        network *m_pnetwork;

    public:
        static void  TrainAndTestNetwork(String^ filename);
    };
}
