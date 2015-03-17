#include "softmax_layer.h"
#include "utility.h"
using namespace std;

softmax_layer::softmax_layer(
    std::shared_ptr<block> input_block,
    std::shared_ptr<block> output_block) {
    this->m_input_block = input_block;
    this->m_output_block = output_block;
}

void softmax_layer::setup_block() {
    if (this->m_output_block->empty()){
        this->m_output_block->resize(this->m_input_block->dims());
    }
    CHECK(cmp_vec(this->m_output_block->dims(), this->m_input_block->dims()));
}

bool softmax_layer::forward(int t) {
    auto& input = this->m_input_block->signal();
    auto& output = this->m_output_block->new_signal();
    softmax_normalize(input, output);
    return true;
}

void softmax_layer::backward(int t) {
    //NOT IMPLEMENT
    CHECK(0);
}

void softmax_layer::end_batch(int size) {
    //NOT IMPLEMENT
    CHECK(0);
}


layer_ptr create_softmax_layer(
    const picojson::value& config, block_factory& bf){

    int input_id = (int) config.get("input").get<double>();
    int output_id = (int) config.get("output").get<double>();
    auto input_block = bf.get_block(input_id);
    auto output_block = bf.get_block(output_id);

    return layer_ptr(new softmax_layer(input_block, output_block));
}

REGISTER_LAYER(softmax, create_softmax_layer);