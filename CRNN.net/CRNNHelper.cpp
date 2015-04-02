#include "CRNNHelper.h"
using namespace CRNNnet;
using namespace std;

string CRNNHelper::MarshalString(String ^ s) {
    using namespace Runtime::InteropServices;
    const char* chars = (const char*) (Marshal::StringToHGlobalAnsi(s)).ToPointer();
    string os = chars;
    Marshal::FreeHGlobal(IntPtr((void*) chars));
    return os;
}
