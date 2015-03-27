#include "gru_layer.h"

gru_layer::gru_layer(block_ptr input, block_ptr output, int output_num, int hide_num) {
    this->m_input_block = input;
    this->m_output_block = output;
    this->m_hide_block = block::new_block();
    this->m_output_num = output_num;
    this->m_hide_num = hide_num;
}

void gru_layer::setup_block() {

}

void gru_layer::setup_params() {

}

bool gru_layer::begin_seq(){

}

bool gru_layer::forward(int t) {

}

void gru_layer::backward(int t) {

}

void gru_layer::end_batch(int size) {

}
