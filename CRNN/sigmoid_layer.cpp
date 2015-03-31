#include "sigmoid_layer.h"
using namespace std;

sigmoid_layer::sigmoid_layer(
    std::shared_ptr<block> input_block,
    std::shared_ptr<block> output_block) {
    this->m_input_block = input_block;
    this->m_output_block = output_block;
}

void sigmoid_layer::setup_block() {
    CHECK(this->m_input_block->size() != 0);
    if (this->m_output_block->size() != 0){
        CHECK(cmp_array_dim(this->m_output_block->signal(), this->m_input_block->signal()));
    }
    else{
        this->m_output_block->resize(this->m_input_block->dims());
    }
    this->m_output_block->error().clear(0);
}


bool sigmoid_layer::forward(int t) {
    auto& output = this->m_output_block->new_signal();
    auto& input = this->m_input_block->signal();

    OMP_FOR
    for (int i = 0; i < input.size(); ++i) {
        output.at(i) = sigmoid(input.at(i));
    }

    this->m_output_history.push_back(output);
    return true;
}

void sigmoid_layer::backward(int t) {
    auto& ierror = this->m_input_block->error();
    auto& oerror = this->m_output_block->error();
    auto& output = this->m_output_history.back();

    OMP_FOR
    for (int i = 0; i < ierror.size(); ++i){
        ierror.at(i) += oerror.at(i) * output.at(i) * (1 - output.at(i));
    }

    oerror.clear(0);
    this->m_output_history.pop_back();
}

bool sigmoid_layer::begin_seq() {
    this->m_output_history.clear();
    this->m_output_block->signal().clear(0);
    this->m_output_block->error().clear(0);
    return true;
}



layer_ptr create_sigmoid_layer(
    const picojson::value& config,
    const string& layer_name,
    block_factory& bf) {
    string input_block_id =  config.get("input").get<string>();
    auto input_block = bf.get_block(input_block_id);
    auto output_block = bf.get_block(layer_name);
    return layer_ptr(new sigmoid_layer(input_block, output_block));
}

REGISTER_LAYER(sigmoid, create_sigmoid_layer);