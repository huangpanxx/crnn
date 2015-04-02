#include "Network.h"
#include "../CRNN/test.h"
#include <msclr/marshal_cppstd.h>
using namespace CRNNnet;
using namespace std;

string MarshalString(String ^ s) {
    using namespace Runtime::InteropServices;
    const char* chars = (const char*) (Marshal::StringToHGlobalAnsi(s)).ToPointer();
    string os = chars;
    Marshal::FreeHGlobal(IntPtr((void*) chars));
    return os;
}

void Network::TrainAndTestNetwork(String^ filename) {
    string stdfilename = MarshalString(filename);
    train_and_test_network(stdfilename);
}

Network::Network(String^ json, String^ plan){
    string sjson = MarshalString(json);
    string splan = MarshalString(plan);
    this->m_pnetwork = new network(sjson, splan);
}

void Network::set_input(const arraykd& data){
    this->m_pnetwork->set_input(data);
}

arraykd Network::forward(){
    return this->m_pnetwork->forward();
}

String^ Network::translate(int k){
    auto ans = this->m_pnetwork->translate(k);
    return msclr::interop::marshal_as<String^>(ans);
}

array<int>^ Network::input_dims() {
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

