#include "pooling_layer.h"

max_pooling_layer::max_pooling_layer(block_ptr input_block, block_ptr output_block, int size){
    this->m_input_block = input_block;
    this->m_output_block = output_block;
    this->m_size = size;
    CHECK(m_size >= 2);
}

bool max_pooling_layer::begin_seq() {
    this->m_max_history.clear();
    return true;
}

void max_pooling_layer::setup_block() {
    CHECK(this->m_input_block->dims().size() == 3);
    auto &ds = this->m_input_block->dims();
    const int rows = (ds[0] + m_size - 1) / m_size,
        cols = (ds[1] + m_size - 1) / m_size,
        channels = ds[2];
    if (m_output_block->empty()){
        m_output_block->resize(rows, cols, channels);
    }
    auto &ods = m_output_block->dims();
    CHECK(ods[0] == rows);
    CHECK(ods[1] == cols);
    CHECK(ods[2] == channels);
}

bool max_pooling_layer::forward(int t) {
    array3d output = m_output_block->new_signal();
    array3d max_index = output.clone(false);
    array3d input = m_input_block->signal();
    int orows = output.rows(), ocols = output.cols(), channels = output.channels();
    int irows = input.rows(), icols = input.cols();

    const int index = m_size * m_size;

    OMP_FOR
    for (int ch = 0; ch < channels; ++ch){
        for (int r = 0; r < orows; ++r) {
            for (int c = 0; c < ocols; ++c) {
                //left top corner
                const int br = r * m_size, bc = c * m_size;

                //first
                float mmax = input.at3(ch, br, bc);
                int k = 0;
                for (int i = 1; i < index; ++i) {
                    const int nr = br + i % m_size, nc = bc + i / m_size;
                    if (nr < irows && nc < icols) {
                        const float val = input.at3(ch, nr, nc);
                        if (val > mmax) {
                            k = i;
                            mmax = val;
                        }
                    }
                }

                //output
                output.at3(ch, r, c) = mmax;

                //record max index
                max_index.at3(ch,r, c) = (float)k;
            }
        }
    }

    this->m_max_history.push_back(max_index);
    return true;
}

void max_pooling_layer::backward(int t) {
    array3d oerror = m_output_block->error();
    array3d ierror = m_input_block->error();
    array3d &max_index = this->m_max_history.back();
    int orows = oerror.rows(), ocols = oerror.cols(), channels = oerror.channels();

    OMP_FOR
    for (int ch = 0; ch < channels; ++ch) {
        for (int r = 0; r < orows; ++r) {
            for (int c = 0; c < ocols; ++c) {
                float err = oerror.at3(ch, r, c);
                int idx = (int)max_index.at3(ch, r, c);
                ierror.at3(ch, r * m_size + idx % m_size, c * m_size + idx / m_size) += err;
            }
        }
    }

    m_max_history.pop_back();
    oerror.clear(0);
}


layer_ptr create_max_pooling_layer(const picojson::value& config,
    block_factory& bf) {
    CHECK(config.contains("input"));
    CHECK(config.contains("output"));
    CHECK(config.contains("size"));
    int input_block_id = (int) config.get("input").get<double>();
    int output_block_id = (int) config.get("output").get<double>();
    auto input_block = bf.get_block(input_block_id);
    auto output_block = bf.get_block(output_block_id);
    int size = (int)config.get("size").get<double>();
    return layer_ptr(new max_pooling_layer(input_block, output_block, size));
}

REGISTER_LAYER(max_pooling, create_max_pooling_layer)
