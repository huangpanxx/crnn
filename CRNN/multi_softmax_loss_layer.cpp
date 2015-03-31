#include "multi_softmax_loss_layer.h"
#include "utility.h"
using namespace std;

multi_softmax_loss_layer::multi_softmax_loss_layer(
    const std::vector<block_ptr> &input_blocks,
    block_ptr &label_block) {
    CHECK(input_blocks.size() > 0);
    this->m_input_blocks = input_blocks;
    this->m_label_block = label_block;
    this->m_loss_num = 0;
    this->m_loss_sum = 0;
}

void multi_softmax_loss_layer::setup_block() {
    CHECK((int)this->m_label_block->dims().size() == 2);
    CHECK(this->m_label_block->dims()[0] >= (int)this->m_input_blocks.size());
    for (auto& block : m_input_blocks) {
        CHECK(block->size() == m_label_block->dims()[1]);
    }
}

void multi_softmax_loss_layer::setup_params() {
    this->m_loss_num = 0;
    this->m_loss_sum = 0;
}

bool multi_softmax_loss_layer::begin_seq() {
    this->m_output_history.clear();
    return true;
}


bool multi_softmax_loss_layer::forward(int t) {
    vector<array> outputs;
    for (auto& block : m_input_blocks){
        auto& input = block->signal();
        auto& output = block->signal().clone(false);
        softmax_normalize(input, output);
        outputs.push_back(output);
    }
    m_output_history.push_back(outputs);

    array2d label = m_label_block->signal();
    const int label_num = (int) m_input_blocks.size();
    const int label_size = label.cols();


    float loss_sum = 0;
#pragma omp parallel for reduction(+:loss_sum)
    for (int i = 0; i < label_num; ++i) {
        auto &output = outputs[i];
        for (int j = 0; j < label_size; ++j) {
            loss_sum -= label.at2(i, j) * (float) log(1e-15 + output.at(j));
        }
    }
    m_loss_num += label_num;
    m_loss_sum += loss_sum;

    return true;
}

void multi_softmax_loss_layer::backward(int t){
    array2d label = m_label_block->signal();
    const int label_num = (int) m_input_blocks.size();
    const int label_size = label.cols();
    const auto &outputs = m_output_history.back();

    OMP_FOR
    for (int i = 0; i < label_num; ++i) {
        auto &err = m_input_blocks[i]->error();
        auto &output = outputs[i];
        for (int j = 0; j < label_size; ++j) {
            err.at(j) = label.at2(i, j) - output.at(j);
        }
    }
    m_output_history.pop_back();
}

float multi_softmax_loss_layer::loss() {
    float floss = 0;
    if (this->m_loss_num != 0) {
        floss = m_loss_sum / m_loss_num;
    }
    return floss;
}

void multi_softmax_loss_layer::end_batch(int t){
    this->m_loss_num = 0;
    this->m_loss_sum = 0;
}

layer_ptr create_multi_softmax_loss_layer(
    const picojson::value& config,
    const string& layer_name,
    block_factory& bf){
    //inputs
    auto input_ids_arr = config.get("inputs").get<picojson::array>();
    vector<string> input_ids;
    for (auto& val : input_ids_arr){
        auto id = val.get<string>();
        input_ids.push_back(id);
    }
    //label
    auto input_blocks = bf.get_blocks(input_ids);
    auto label_block = bf.get_block("label");
    return layer_ptr(new multi_softmax_loss_layer(input_blocks, label_block));
}

REGISTER_LAYER(multi_softmax_loss, create_multi_softmax_loss_layer);