#pragma once
#include "../CRNN/common.h"
#include "../CRNN/memory.h"

namespace CRNNnet {
    public ref class FloatArray
    {
    public:
        FloatArray(const FloatArray^ &copier);
        FloatArray(array<int>^ dims, array<float>^ data);
        FloatArray(const arraykd& arr);
        ~FloatArray();
        arraykd* Array();
        float At(int i){
            return this->m_array->at(i);
        }
        float At2(int r, int c){
            return ((array2d*)this->m_array)->at2(r, c);
        }
        float At3(int ch, int r, int c){
            return ((array3d*)this->m_array)->at3(ch, r, c);
        }
        int rows(){
            return m_array->dim(0);
        }
        int cols(){
            return m_array->dim(1);
        }
        int channels(){
            return  m_array->dim(2);
        }
        int size(){
            return m_array->size();
        }
        int arg_max(){
            return this->m_array->arg_max();
        }
    private:
        arraykd* m_array;
    };
}
