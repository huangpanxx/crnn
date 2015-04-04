#include "multi_softmax_layer.h"
#include "utility.h"

using namespace std;

multi_softmax_layer::multi_softmax_layer(
    const vector<block_ptr> &input_blocks,
    block_ptr &output_block){
    this->m_input_blocks = input_blocks;
    this->m_output_block = output_block;
    CHECK(input_blocks.size() > 0);
}

void multi_softmax_layer::setup_block(){
    const int rows = (int)this->m_input_blocks.size();
    const int cols = (int)this->m_input_blocks[0]->size();
    //must have the same size
    for (auto &block : m_input_blocks){
        CHECK(block->size() == cols);
    }
    //init output block
    if (this->m_output_block->empty()) {
        this->m_output_block->resize(rows, cols);
    }
    CHECK(this->m_output_block->dims()[0] == rows);
    CHECK(this->m_output_block->dims()[1] == cols);
}

bool multi_softmax_layer::forward() {
    array2d output = this->m_output_block->new_signal();

    OMP_FOR
    for (int i = 0; i < (int) m_input_blocks.size(); ++i){
        auto& input = m_input_blocks[i]->signal();
        softmax_normalize(input, output, i);
    }
    return true;
}

void multi_softmax_layer::backward() {
    //NOT IMPLEMENT
    CHECK(0);
}

void multi_softmax_layer::end_batch(int size) {
    //NOT IMPLEMENT
    CHECK(0);
}


layer_ptr create_multi_softmax_layer(
    const picojson::value& config,
    const string& layer_name,
    block_factory& bf){
    auto input_ids_arr = config.get("inputs").get<picojson::array>();
    vector<string> input_ids;
    for (auto &val : input_ids_arr){
        auto id =  val.get<string>();
        input_ids.push_back(id);
    }
    auto input_blocks = bf.get_blocks(input_ids);
    auto output_block = bf.get_block(layer_name);
    return layer_ptr(new multi_softmax_layer(input_blocks, output_block));
}

REGISTER_LAYER(multi_softmax, create_multi_softmax_layer)