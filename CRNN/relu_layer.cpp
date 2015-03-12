#include "relu_layer.h"
#include "utility.h"
using namespace std;

relu_layer::relu_layer(
    std::shared_ptr<block> input_block,
    std::shared_ptr<block> output_block,
    float negtive_slope) {
    this->m_input_block = input_block;
    this->m_output_block = output_block;
    this->m_negtive_slop = negtive_slope;
}

void  relu_layer::setup_block() {
    CHECK(this->m_input_block->size() != 0);
    if (this->m_output_block->size() != 0){
        CHECK(cmp_array_dim(this->m_output_block->signal(), this->m_input_block->signal()));
    }
    else{
        this->m_output_block->resize(this->m_input_block->dims());
    }

}


bool relu_layer::forward(int t) {
    auto& output = this->m_output_block->new_signal();
    auto& input = this->m_input_block->signal();
    int size = input.size();

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output.at(i) = relu(input.at(i));
    }

    this->m_input_history.push_back(input);

    return true;
}

void relu_layer::backward(int t) {
    auto& ierror = this->m_input_block->error();
    auto& oerror = this->m_output_block->error();
    auto& input = this->m_input_history.back();
    int size = ierror.size();

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        ierror.at(i) += oerror.at(i) *  rrelu(input.at(i));
    }

    this->m_input_history.pop_back();
    oerror.clear(0);
}

bool relu_layer::begin_seq() {
    this->m_input_history.clear();
    this->m_output_block->signal().clear(0);
    this->m_output_block->error().clear(0);
    return true;
}

void relu_layer::save(std::ostream& os) {
    write_val_to_stream(os, this->m_negtive_slop);
}

void relu_layer::load(std::istream& is) {
    read_val_from_stream(is, this->m_negtive_slop);
}

layer_ptr create_relu_layer(const picojson::value& config,
    block_factory& bf) {
    float slope = (float)config.get("negtive_slope").get<double>();
    int input_block_id = (int)config.get("input").get<double>();
    int output_block_id = (int) config.get("output").get<double>();
    auto input_block = bf.get_block(input_block_id);
    auto output_block = bf.get_block(output_block_id);
    return layer_ptr(new relu_layer(input_block, output_block, slope));
}

REGISTER_LAYER(relu, create_relu_layer);