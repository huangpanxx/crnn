#include "Network.h"
#include "../CRNN/test.h"
#include "CRNNHelper.h"
using namespace CRNNnet;
using namespace std;
using namespace System;


void Network::TrainAndTestNetwork(String^ filename) {
    string stdfilename = CRNNHelper::MarshalString(filename);
    train_and_test_network(stdfilename);
}

Network::Network(String^ json, String^ plan){
    string sjson = CRNNHelper::MarshalString(json);
    string splan = CRNNHelper::MarshalString(plan);
    this->m_pnetwork = new network(sjson, splan);
}

void Network::SetInput(FloatArray^ data){
    auto &arr = *data->Array();
    this->m_pnetwork->set_input(arr);
}

FloatArray^ Network::Forward(){
    arraykd arr = this->m_pnetwork->forward();
    return gcnew FloatArray(arr);
}

String^ Network::Translate(int k){
    auto ans = this->m_pnetwork->translate(k);
    return gcnew String(ans.c_str());
}

array<int>^ Network::InputDims() {
    auto dims = this->m_pnetwork->input_dims();
    auto arr = gcnew array<int>((int)dims.size());
    for (int i = 0; i < (int) dims.size(); ++i){
        arr[i] = dims[i];
    }
    return arr;
}

Network::~Network() {
    delete this->m_pnetwork;
}

