#include "conv_layer.h"
#include "utility.h"
using namespace std;

conv_layer::conv_layer(
    block_ptr input_block,
    block_ptr output_block,
    int kernel_size,
    int kernel_num,
    int kernel_stride) {
    CHECK(kernel_stride >= 0);
    CHECK(kernel_stride < kernel_size);
    CHECK(kernel_num >= 1);

    this->m_input_block = input_block;
    this->m_output_block = output_block;
    this->m_kernel_size = kernel_size;
    this->m_kernel_num = kernel_num;
    this->m_kernel_stride = kernel_stride;
}

void conv_layer::setup_block() {
    auto& idims = m_input_block->dims();
    CHECK(idims.size() == 3);
    int rows = idims[0], cols = idims[1], channels = idims[2];
    CHECK(rows >= m_kernel_size && cols >= m_kernel_size);

    //output map size
    int nrows = (rows - m_kernel_size) / m_kernel_stride + 1;
    int ncols = (cols - m_kernel_size) / m_kernel_stride + 1;

    if (m_output_block->empty()){
        //resize output block
        m_output_block->resize(nrows, ncols, m_kernel_num);
    }
    else{
        //check output block
        auto& odims = m_output_block->dims();
        CHECK(odims[0] == nrows);
        CHECK(odims[1] == ncols);
        CHECK(odims[2] == m_kernel_num);
    }
    m_output_block->error().clear(0);
}

void conv_layer::setup_params() {
    int channels = m_input_block->dims()[2];

    float bound = sqrtf(6.0f / (m_input_block->size() + m_output_block->size()));

    //weights
    if (m_weights.size() == 0){
        m_weights = array4d(m_kernel_size, m_kernel_size, channels, m_kernel_num);
        m_weights.rand(-bound, bound);
    }

    CHECK(m_weights.rows() == m_kernel_size);
    CHECK(m_weights.cols() == m_kernel_size);
    CHECK(m_weights.channels() == channels);
    CHECK(m_weights.nums() == m_kernel_num);

    //bias
    if (m_bias.size() == 0) {
        m_bias = arraykd(m_kernel_num);
        m_bias.rand(0.01f, 0.5f);
    }
    else{
        CHECK(m_bias.size() == m_kernel_num);
    }

    //weights grad
    this->m_grad_weights = m_weights.clone(false);
    this->m_grad_weights.clear(0);

    //bias grad
    m_grad_bias = m_bias.clone(false);
    m_grad_bias.clear(0);
}

bool conv_layer::begin_seq() { 
    this->m_input_history.clear();
    return true; 
}

bool conv_layer::forward() { 
    array3d input = m_input_block->signal();
    array3d output = m_output_block->new_signal();

    int rows = input.rows(), cols = input.cols(), channels = input.channels();
    const int orows = output.rows(), ocols = output.cols(), ochannels = m_weights.nums();

    //for each output point
    OMP_FOR
    for (int r = 0; r < orows; ++r) {
        for (int c = 0; c < ocols; ++c) {
            for (int och = 0; och < ochannels; ++och){
                float bias = m_bias.at(och);
                float &output_unit = output.at3(r, c, och);
                output_unit = bias;
                int y = r * m_kernel_stride, x = c * m_kernel_stride;
                //for input region
                for (int dr = 0; dr < m_kernel_size; ++dr){
                    for (int dc = 0; dc < m_kernel_size; ++dc) {
                        for (int ch = 0; ch < channels; ++ch){
                            int nr = y + dr, nc = x + dc;
                            output_unit += input.at3(nr, nc, ch) * m_weights.at4(och, dr, dc, ch);
                        }
                    }
                }
            }
        }
    }
    this->m_input_history.push_back(input);
    return true; 
};

void conv_layer::backward() {
    array3d input = this->m_input_history.back();
    array3d oerror = this->m_output_block->error();
    array3d ierror = this->m_input_block->error();


    if (this->enable_bp()) {
        const int rows = input.rows(), cols = input.cols(), channels = input.channels();
        CHECK(m_weights.nums() == oerror.channels());
        const int orows = oerror.rows(), ocols = oerror.cols(), ochannels = oerror.channels();

        //for output point
        OMP_FOR
        for (int r = 0; r < orows; ++r) {
            for (int c = 0; c < ocols; ++c) {
                for (int och = 0; och < ochannels; ++och) {
                    auto& gb = m_grad_bias.at(och);
                    float err = oerror.at3(r, c, och);
                    gb += err;
                    int y = r * m_kernel_stride, x = c * m_kernel_stride;
                    for (int dr = 0; dr < m_kernel_size; ++dr) {
                        for (int dc = 0; dc < m_kernel_size; ++dc) {
                            for (int ich = 0; ich < channels; ++ich){
                                int nr = y + dr, nc = x + dc;
                                ierror.at3(nr, nc, ich) += err * m_weights.at4(och, dr, dc, ich);
                                m_grad_weights.at4(och, dr, dc,ich) += err * input.at3(nr, nc, ich);
                            }
                        }
                    }
                }
            }
        }
    }

    m_input_history.pop_back();
    oerror.clear(0);
};

void conv_layer::end_batch(int size) {
    float md = this->momentum_decay();
    float lr = this->learn_rate() / size;

    m_weights.mul_add(m_grad_weights, lr / (1.0f + md));
    m_grad_weights.mul(md / (1.0f + md));

    m_bias.mul_add(m_grad_bias, lr);
    m_grad_bias.mul(md / (1.0f + md));
};



void conv_layer::save(std::ostream& is){
    write_val_to_stream(is, this->m_kernel_num);
    write_val_to_stream(is, this->m_kernel_size);
    write_val_to_stream(is, this->m_kernel_stride);
    write_array_to_stream(is, m_weights);
    write_array_to_stream(is, this->m_bias);
}

void conv_layer::load(std::istream& os){
    read_val_from_stream(os, this->m_kernel_num);
    read_val_from_stream(os, this->m_kernel_size);
    read_val_from_stream(os, this->m_kernel_stride);
    this->m_weights = read_array_from_stream(os);
    this->m_bias = read_array_from_stream(os);
}

layer_ptr create_conv_layer(
    const picojson::value& config,
    const string& layer_name,
    block_factory& bf) {
    auto name = config.get("name").get<string>();
    auto input_block_id = config.get("input").get<string>();
    int kernel_size = (int)config.get("kernel_size").get<double>();
    int kernel_num = (int)config.get("kernel_num").get<double>();
    int kernel_stride = (int)config.get("kernel_stride").get<double>();
    CHECK(kernel_size > 0);
    CHECK(kernel_num > 0);
    auto input_block = bf.get_block(input_block_id);
    auto output_block = bf.get_block(layer_name);
    return layer_ptr(new conv_layer(input_block, output_block,
        kernel_size, kernel_num, kernel_stride));
    return 0;
}

REGISTER_LAYER(conv, create_conv_layer);