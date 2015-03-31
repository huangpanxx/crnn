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
    this->m_mid_history.clear();
    this->m_output_block->error().clear(0);
    this->m_output_block->signal().clear(0);
    return true; 
}

bool tanh_layer::forward(int t){
    auto& input = this->m_input_block->signal();
    auto& output = this->m_output_block->new_signal();
    auto mid = input.clone(false);

    const int size = input.size();

    OMP_FOR
    for (int i = 0; i < size; ++i){
        float x = input.at(i);
        float q = x>0 ? exp(-x) : exp(x);
        float v = q*q;
        if (x < 0) { v = (v - 1) / (v + 1); }
        else{ v = (1 - v) / (v + 1); }
        output.at(i) = v;
        mid.at(i) = q;
    }

    m_mid_history.push_back(mid);

    return true;
};

void tanh_layer::backward(int t){
    auto& mid = this->m_mid_history.back();
    auto& ierr = this->m_input_block->error();
    auto& oerr = this->m_output_block->error();
    const int size = ierr.size();

    OMP_FOR
    for (int i = 0; i < size; ++i){
        float q = mid.at(i);
        float v = 2 * q / (1 + q*q);
        ierr.at(i) += oerr.at(i) * v*v;
    }

    oerr.clear(0);

    this->m_mid_history.pop_back();
};