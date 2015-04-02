#include "Utility.h"

using namespace CRNNnet;
using namespace std;

void fill(arraykd& arr, array<float> ^data){
    const int sz = arr.size();
    for (int i = 0; i < sz; ++i){
        arr.at(i) = data[i];
    }
}

arraykd Utility::CreateArray(int size, array<float>^ data){
    auto arr = arraykd(size);
    fill(arr, data);
    return arr;
}

array2d Utility::CreateArray(int rows, int cols, array<float>^ data){
    auto arr = array2d(rows, cols);
    fill(arr, data);
    return arr;
}

array3d Utility::CreateArray(int channel, int rows, int cols, array<float>^ data){
    auto arr = array3d(rows, cols, channel);
    fill(arr, data);
    return arr;
}
