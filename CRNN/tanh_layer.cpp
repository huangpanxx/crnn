#include "tanh_layer.h"
using namespace std;

tanh_layer::tanh_layer(block_ptr input, block_ptr output) {
    this->m_input_block = input;
    this->m_output_block = output;
}

void tanh_layer::setup_block(){
    if (m_output_block->empty()){
        m_output_block->resize(this->m_input_block->dims());
    }
    CHECK(cmp_vec(m_input_block->dims(), m_output_block->dims()));
};

bool tanh_layer::begin_seq(){ 
    return true; 
}

bool tanh_layer::forward(int t){
    return true;
};

void tanh_layer::backward(int t){

};