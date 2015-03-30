#include "inner_product_layer.h"
#include "utility.h"
using namespace std;

inner_product_layer::inner_product_layer(
    const vector<shared_ptr<block> > &input_blocks,
    const shared_ptr<block> &output_block,
    int output_num) {
    initialize(input_blocks, output_block, output_num);
}

void inner_product_layer::initialize(
    const std::vector<std::shared_ptr<block> > &input_blocks,
    const std::shared_ptr<block> &output_block,
    int output_num){
    CHECK(input_blocks.size() > 0);
    CHECK(output_num > 0);
    this->m_output_num = output_num;
    this->m_input_blocks = input_blocks;
    this->m_output_block = output_block;
}

inner_product_layer::inner_product_layer(
    const std::shared_ptr<block> &input_block,
    const std::shared_ptr<block> &output_block,
    int output_num) {
    vector<shared_ptr<block> > input_blocks;
    input_blocks.push_back(input_block);
    initialize(input_blocks, output_block, output_num);
}

void inner_product_layer::setup_block() {
    if (this->m_output_block->empty()) {
        this->m_output_block->resize(this->m_output_num);
    }
    else {
        CHECK(this->m_output_block->size() == this->m_output_num);
    }
    this->m_output_block->error().clear(0);
}

void inner_product_layer::setup_params(){
    //bias
    if (this->m_bias.size() == 0) {
        this->m_bias = array(m_output_num);
        this->m_bias.rand(-0.5f, 0.5f);
    }
    else {
        CHECK(this->m_bias.size() == m_output_num);
    }

    //bias grad
    this->m_grad_bias = this->m_bias.clone(false);
    this->m_grad_bias.clear(0);

    //weights
    for (int i = 0; i < (int) m_input_blocks.size(); ++i) {
        //weight
        auto& block = this->m_input_blocks[i];
        float bound = sqrtf(6.0f / (block->size() + m_output_block->size()));
        if ((int)this->m_weights.size() > i){
            auto& w = m_weights[i];
            CHECK(w.rows() == m_output_num);
            CHECK(w.cols() == block->size());
        }
        else {
            array2d w(m_output_num, block->size());
            w.rand(-bound, bound);
            this->m_weights.push_back(w);
        }
        //weight grad
        this->m_grad_weights.push_back(m_weights[i].clone(false));
        this->m_grad_weights.back().clear(0);
    }
}

bool inner_product_layer::forward(int t) {
    //input signals
    std::vector<array> inputs;
    for (auto &arr : m_input_blocks){
        inputs.push_back(arr->signal());
    }

    //back inputs
    m_inputs_history.push_back(inputs);

    //new output
    auto &output = this->m_output_block->new_signal();

    //bias
    output.copy(m_bias);

    //output
    for (int i = 0; i < (int) inputs.size(); ++i){
        auto& input = inputs[i];
        auto& weight = m_weights[i];
        mul_addv(weight, input, output);
    }

    return true;
}

void inner_product_layer::backward(int t) {
    auto &inputs = m_inputs_history.back();
    auto &error = m_output_block->error();

    //bp error to input blocks
    if (this->enable_bp()) {
        for (int i = 0; i < (int) inputs.size(); ++i) {
            auto& input_block = m_input_blocks[i];
            auto& w = m_weights[i];
            auto& ierror = input_block->error();
            mul_addh(error, w, ierror);
        }
    }

    //grad bias
    this->m_grad_bias.add(error);

    //grad  weights
    for (int i = 0; i < (int) inputs.size(); ++i) {
        auto& input = inputs[i];
        auto& gw = m_grad_weights[i];
        int esz = error.size(), isz = input.size();
        OMP_FOR
        for (int j = 0; j < esz; ++j) {
            for (int k = 0; k < isz; ++k) {
                gw.at2(j, k) += input.at(k) * error.at(j);
            }
        }
    }

    //pop inputs history
    error.clear(0);
    m_inputs_history.pop_back();
}

void inner_product_layer::end_batch(int size) {
    float md = this->momentum_decay();
    float lr = this->learn_rate() / size;

    for (int i = 0; i < (int) m_weights.size(); ++i){
        auto& grad = m_grad_weights[i];
        m_weights[i].mul_add(grad, lr / (1.0f + md));
        grad.mul(md / (md + 1.0f));
    }

    m_bias.mul_add(m_grad_bias, lr);
    m_grad_bias.mul(md / (md + 1.0f));
}

bool inner_product_layer::begin_seq() {
    this->m_inputs_history.clear();
    m_output_block->signal().clear(0);
    m_output_block->error().clear(0);
    return true;
}

void inner_product_layer::save(std::ostream& os) {
    write_val_to_stream(os, this->m_output_num);
    write_arrays_to_stream(os, convert_arrays<array>(this->m_weights));
    write_array_to_stream(os, this->m_bias);
}

void inner_product_layer::load(std::istream& is) {
    read_val_from_stream(is, this->m_output_num);
    this->m_weights = convert_arrays<array2d>(read_arrays_from_stream(is));
    this->m_bias = read_array_from_stream(is);
}

layer_ptr create_inner_product_layer(
    const picojson::value& config,
    block_factory& bf) {
    auto inputs = config.get("inputs").get<picojson::array>();
    vector<int> input_ids;
    for (auto input : inputs) {
        int id = (int) input.get<double>();
        input_ids.push_back(id);
    }
    sort(input_ids.begin(), input_ids.end());
    vector<block_ptr> input_blocks = bf.get_blocks(input_ids);

    auto output_id = (int) config.get("output").get<double>();
    auto output_block = bf.get_block(output_id);

    int output_num = (int) config.get("output_num").get<double>();

    return layer_ptr(new inner_product_layer(input_blocks, output_block, output_num));
}

REGISTER_LAYER(inner_product, create_inner_product_layer);
