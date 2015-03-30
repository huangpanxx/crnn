#include "wise_product_layer.h"
using namespace std;

wise_product_layer::wise_product_layer(block_ptr input1, block_ptr input2, block_ptr output) {
    this->m_input_block1 = input1;
    this->m_input_block2 = input2;
    this->m_output_block = output;
}

void wise_product_layer::setup_block() {
    CHECK(this->m_input_block1->size() > 0);
    CHECK(this->m_input_block1->size() == this->m_input_block2->size());
    this->m_output_block->resize(this->m_input_block1->size());
}

bool wise_product_layer::begin_seq() {
    this->m_output_block->error().clear();
    this->m_output_block->signal().clear(0);
    this->m_input_history.clear();
    return true;
}

bool wise_product_layer::forward(int t){
    auto& input1 = this->m_input_block1->signal();
    auto& input2 = this->m_input_block2->signal();
    auto& output = this->m_output_block->new_signal();
    mul_wise(input1, input2, output);
    this->m_input_history.push_back(make_pair(input1, input2));
    return true;
}

void wise_product_layer::backward(int t){
    auto& pair = this->m_input_history.back();

    auto& input1 = pair.first;
    auto& input2 = pair.second;
    auto& err1 = m_input_block1->error();
    auto& err2 = m_input_block2->error();
    auto& oerr =  this->m_output_block->error();

    mul_wise_add(oerr, input2, err1);
    mul_wise_add(oerr, input1, err2);
    oerr.clear(0);

    m_input_history.pop_back();
}
