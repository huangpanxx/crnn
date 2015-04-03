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
        int Rows(){
            return m_array->dim(0);
        }
        int Cols(){
            return m_array->dim(1);
        }
        int Channels(){
            return  m_array->dim(2);
        }
        int Size(){
            return m_array->size();
        }
        int ArgMax(){
            return this->m_array->arg_max();
        }
    private:
        arraykd* m_array;
    };
}
