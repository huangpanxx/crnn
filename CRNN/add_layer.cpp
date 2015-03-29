#include "add_layer.h"

add_layer::add_layer(
    const std::vector<block_ptr> &input_blocks,
    block_ptr &output_block) {
    this->m_input_blocks = input_blocks;
    this->m_output_block = output_block;
    CHECK(this->m_input_blocks.size() > 0);
}

void add_layer::setup_block(){
    auto dims = this->m_input_blocks[0]->dims();
    for (int i = 1; i < (int)m_input_blocks.size(); ++i){
        CHECK(cmp_vec(dims, this->m_input_blocks[i]->dims()));
    }
    m_output_block->resize(dims);
}

bool add_layer::forward(int t){
    auto& output = this->m_output_block->new_signal();
    output.clear(0);
    for (int j = 0; j < (int)this->m_input_blocks.size(); ++j){
        auto &input = m_input_blocks[j]->signal();
        output.add(input);
    }
}

void add_layer::backward(int t){
    auto& oerr = this->m_output_block->error();
    for (int i = 0; i < this->m_input_blocks.size(); ++i){
        auto& ierr = m_input_blocks[i]->error();
        ierr.copy(oerr);
    }
}

bool add_layer::begin_seq(){
    this->m_output_block->error().clear(0);
    this->m_output_block->signal().clear(0);
}