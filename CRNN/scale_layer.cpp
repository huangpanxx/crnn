#include "scale_layer.h"

scale_layer::scale_layer(block_ptr input, block_ptr output,
    float scale,float bias) {
    this->m_input_block = input;
    this->m_output_block = output;
    this->m_scale = scale;
    this->m_bias = bias;
}

void scale_layer::setup_block() {
    CHECK(this->m_input_block->size() > 0);
    this->m_output_block->resize(this->m_input_block->size());
}

bool scale_layer::begin_seq() {
    this->m_output_block->signal().clear(0);
    this->m_output_block->error().clear(0);
    return true;
}

bool scale_layer::forward(int t) {
    auto& output = this->m_output_block->new_signal();
    auto& input = this->m_input_block->signal();
    if (m_bias != 0){
        mul(input, m_scale, output, m_bias);
    }
    else {
        mul(input, m_scale, output);
    }
    return true;
}

void scale_layer::backward(int t) {
    auto& oerr = this->m_output_block->error();
    auto& ierr = this->m_input_block->error();
    mul_add(oerr, m_scale, ierr);
    oerr.clear(0);
}
