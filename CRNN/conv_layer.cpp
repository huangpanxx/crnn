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
    for (int i = 0; i < this->m_kernel_num; ++i){
        if ((int)this->m_weights.size() <= i) {
            auto& w = array3d(m_kernel_size, m_kernel_size, channels);
            w.rand(-bound, bound);
            this->m_weights.push_back(w);
        }
        else{
            auto& w = this->m_weights[i];
            CHECK(w.rows() == m_kernel_size);
            CHECK(w.cols() == m_kernel_size);
            CHECK(w.channels() == channels);
        }
    }

    //bias
    if (m_bias.size() == 0) {
        m_bias = array(m_kernel_num);
        m_bias.rand(0.01f, 0.5f);
    }
    else{
        CHECK(m_bias.size() == m_kernel_num);
    }

    //weights grad
    for (int i = 0; i < (int)m_weights.size(); ++i){
        array gw = m_weights[i].clone(false);
        gw.clear(0);
        this->m_grad_weights.push_back(gw);
    }

    //bias grad
    m_grad_bias = m_bias.clone(false);
    m_grad_bias.clear(0);
}

bool conv_layer::begin_seq() { 
    this->m_input_history.clear();
    return true; 
}

bool conv_layer::forward(int t) { 
    array3d input = m_input_block->signal();
    array3d output = m_output_block->new_signal();

    int rows = input.rows(), cols = input.cols(), channels = input.channels();
    const int orows = output.rows(), ocols = output.cols(), ochannels = (int)m_weights.size();

    //for each output point
    #pragma omp parallel for
    for (int och = 0; och < ochannels; ++och){
        auto& w = m_weights[och];
        float bias = m_bias.at(och);
        for (int r = 0; r < orows ; ++r) {
            for (int c = 0; c < ocols ; ++c) {
                float &output_unit = output.at3(och, r, c);
                output_unit = bias;

                //for input region
                for (int ch = 0; ch < channels; ++ch){
                    for (int dr = 0; dr < m_kernel_size; ++dr){
                        for (int dc = 0; dc < m_kernel_size; ++dc) {
                            int nr = r * m_kernel_stride + dr;
                            int nc = c * m_kernel_stride + dc;
                            if (nr < rows && nc < cols) {
                                output_unit += input.at3(ch, nr, nc) * w.at3(ch, dr, dc);
                            }
                        }
                    }
                }
            }
        }
    }
    this->m_input_history.push_back(input);
    return true; 
};

void conv_layer::backward(int t) {
    array3d input = this->m_input_history.back();
    array3d oerror = this->m_output_block->error();
    array3d ierror = this->m_input_block->error();


    const int rows = input.rows(), cols = input.cols(), channels = input.channels();
    CHECK(m_weights.size() == oerror.channels());
    const int orows = oerror.rows(), ocols = oerror.cols(), ochannels = oerror.channels();

    //for output point
    #pragma omp parallel for
    for (int och = 0; och < ochannels; ++och) {
        auto& w = m_weights[och];
        auto& gw = m_grad_weights[och];
        auto& gb = m_grad_bias.at(och);

        for (int r = 0; r < orows; ++r) {
            for (int c = 0; c < ocols; ++c) {
                float err = oerror.at3(och, r, c);

                //for input region
                gb += err;
                for (int ich = 0; ich < channels; ++ich){
                    for (int dr = 0; dr < m_kernel_size; ++dr) {
                        for (int dc = 0; dc < m_kernel_size; ++dc) {
                            int nr = r * m_kernel_stride + dr;
                            int nc = c * m_kernel_stride + dc;
                            if (nr < rows && nc < cols) {
                                //bp
                                ierror.at3(ich, nr, nc) += err * w.at3(ich, dr, dc);
                                //weight grad
                                gw.at3(ich, dr, dc) += err * input.at3(ich, nr, nc);
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

    for (int i = 0; i < (int) m_weights.size(); ++i){
        auto& grad = m_grad_weights[i];
        m_weights[i].mul_add(grad, lr / (1.0f + md));
        grad.mul(md / (md + 1.0f));
    }

    m_bias.mul_add(m_grad_bias, lr);
    m_grad_bias.mul(md / (md + 1.0f));;
};



void conv_layer::save(std::ostream& is){
    write_val_to_stream(is, this->m_kernel_num);
    write_val_to_stream(is, this->m_kernel_size);
    write_val_to_stream(is, this->m_kernel_stride);
    auto weights = convert_arrays<array>(this->m_weights);
    write_arrays_to_stream(is, weights);
    write_array_to_stream(is, this->m_bias);
}

void conv_layer::load(std::istream& os){
    read_val_from_stream(os, this->m_kernel_num);
    read_val_from_stream(os, this->m_kernel_size);
    read_val_from_stream(os, this->m_kernel_stride);
    this->m_weights = convert_arrays<array3d>(read_arrays_from_stream(os));
    this->m_bias = read_array_from_stream(os);
}


layer_ptr create_conv_layer(
    const picojson::value& config,
    block_factory& bf) {
    auto name = config.get("name").get<string>();
    int input_block_id = (int)config.get("input").get<double>();
    int output_block_id = (int)config.get("output").get<double>();
    int kernel_size = (int)config.get("kernel_size").get<double>();
    int kernel_num = (int)config.get("kernel_num").get<double>();
    int kernel_stride = (int)config.get("kernel_stride").get<double>();
    CHECK(kernel_size > 0);
    CHECK(kernel_num > 0);
    auto input_block = bf.get_block(input_block_id);
    auto output_block = bf.get_block(output_block_id);
    return layer_ptr(new conv_layer(input_block, output_block,
        kernel_size, kernel_num, kernel_stride));
    return 0;
}

REGISTER_LAYER(conv, create_conv_layer);