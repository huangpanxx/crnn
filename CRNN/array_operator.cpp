#include "array_operator.h"
#include <amp.h>
using namespace std;
using namespace concurrency;



array_operator_ptr get_default_array_operator(){
    static array_operator_ptr ptr(0);
    //this var is for debug
#ifdef _DEBUG
    const bool try_use_gpu = true;
#else
    const bool try_use_gpu = true;
#endif

    if (!ptr){
        auto accname = concurrency::accelerator::default_accelerator;
        static auto acc = concurrency::accelerator(accname);
        if (acc.is_emulated || !try_use_gpu){
            cout << "cpu mode" << endl;
            ptr = array_operator_ptr(new array_operator());
        }
        else{
            wcout << acc.description << " detected!" << endl;
            ptr = array_operator_ptr(new array_operator(&acc.default_view));
        }
    }
    return ptr;
}

