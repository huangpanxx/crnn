#include "relu_layer.h"
#include "utility.h"
#include "network.h"
using namespace std;

relu_layer::relu_layer(
    std::shared_ptr<block> input_block,
    std::shared_ptr<block> output_block,
    bool share) {
    this->m_input_block = input_block;
    this->m_output_block = output_block;
    this->m_share = share;
}

void  relu_layer::setup_block(){
    CHECK(this->m_input_block->size() != 0);
    if (this->m_output_block->size() != 0){
        CHECK(cmp_vec(this->m_output_block->dims(), this->m_input_block->dims()));
    }
    else{
        this->m_output_block->resize(this->m_input_block->dims());
    }
}

void relu_layer::setup_params() {
    const int sz = m_share ? 1 : this->m_input_block->size();
    if (m_negtive_slop.size() == 0){
        this->m_negtive_slop = arraykd(sz);
        this->m_negtive_slop.clear(-1.0f);
    }
    CHECK(m_negtive_slop.size() == sz);
    this->m_negtive_slop_grad = m_negtive_slop.clone(false);
    this->m_negtive_slop_grad.clear(0);
}


bool relu_layer::forward() {
    auto& output = this->m_output_block->new_signal();
    auto& input = this->m_input_block->signal();
    const int size = input.size();

    OMP_FOR
    for (int i = 0; i < size; ++i) {
        float val = input.at(i);
        if (val < 0) {
            int k = m_share ? 0 : i;
            val *= m_negtive_slop.at(k);
        }
        output.at(i) = val;
    }

    this->m_input_history.push_back(input);
    return true;
}

void relu_layer::backward() {
    auto& ierror = this->m_input_block->error();
    auto& oerror = this->m_output_block->error();
    auto& input = this->m_input_history.back();
    const int size = ierror.size();

    CHECK(ierror.size() == oerror.size());

    OMP_FOR
    for (int i = 0; i < size; ++i) {
        float val = input.at(i);
        float err = oerror.at(i);
        if (val < 0) {
            int k = m_share ? 0 : i;
            m_negtive_slop_grad.at(k) += err * val;
            err *= m_negtive_slop.at(k);
        }
        ierror.at(i) += err;
    }

    oerror.clear(0);
    this->m_input_history.pop_back();
}

bool relu_layer::begin_seq() {
    this->m_output_block->signal().clear(0);
    this->m_output_block->error().clear(0);
    this->m_input_history.clear();
    return true;
}

void relu_layer::save(std::ostream& os) {
    write_array_to_stream(os, this->m_negtive_slop);
}

void relu_layer::load(std::istream& is) {
    this->m_negtive_slop = read_array_from_stream(is);
}

void relu_layer::end_batch(int size) {
    float md = this->momentum_decay();
    float lr = this->learn_rate() / size;
    m_negtive_slop.mul_add(m_negtive_slop_grad, lr / (1.0f + md));
    m_negtive_slop_grad.mul(md / (md + 1.0f));
}

layer_ptr create_relu_layer(
    const picojson::value& config,
    const string& layer_name,
    network* net) {
    CHECK(config.contains("input"));
    auto input_block_id = config.get("input").get<string>();
    auto& input_block = net->block(input_block_id);
    auto& output_block = net->block(layer_name);
    bool share = false;
    if (config.contains("share")){
        share = config.get("share").get<bool>();
    }
    return layer_ptr(new relu_layer(input_block, output_block, share));
}

REGISTER_LAYER(relu, create_relu_layer);
