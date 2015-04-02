// 这是主 DLL 文件。

#include "CRNN.net.h"
#include "../CRNN/test.h"

using namespace CRNNnet;
using namespace std;

static int DUMMY_VAR = CRNN_INITIALIZER;

void MarshalString(String ^ s, string& os) {
    using namespace Runtime::InteropServices;
    const char* chars = (const char*) (Marshal::StringToHGlobalAnsi(s)).ToPointer();
    os = chars;
    Marshal::FreeHGlobal(IntPtr((void*) chars));
}

void Network::TestNetwork(String^ filename) {
    string stdfilename;
    MarshalString(filename, stdfilename);
    //MarshalString(filename, stdfilename);
    test_network(stdfilename);
}
