#include "FloatArray.h"
using namespace std;
using namespace CRNNnet;

FloatArray::FloatArray(array<int>^ dims, array<float>^ data){
    vector<int> _dims;
    for (int i = 0; i < dims->Length; ++i){
        int k = dims[i];
        _dims.push_back(k);
    }
    this->m_array = new arraykd(_dims);
    for (int i = 0; i < data->Length; ++i){
        this->m_array->at(i) = data[i];
    }
}

FloatArray::FloatArray(const FloatArray^ &copier){
    if (copier != this){
        delete this->m_array;
        auto arr = copier->m_array;
        this->m_array = new arraykd(arr->dims());
        this->m_array->copy(*arr);
    }
}

FloatArray::FloatArray(const arraykd& arr){
    this->m_array = new arraykd(arr.dims());
    this->m_array->copy(arr);
}

FloatArray::~FloatArray(){
    delete m_array;
}

arraykd* FloatArray::Array(){
    return  this->m_array;
}
