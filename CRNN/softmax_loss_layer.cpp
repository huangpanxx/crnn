#include "softmax_loss_layer.h"
using namespace std;

softmax_loss_layer::softmax_loss_layer(
    shared_ptr<block> input_block,
    shared_ptr<block> label_block) {
    this->m_input_block = input_block;
    this->m_label_block = label_block;
    this->m_report = false;
}

void softmax_loss_layer::setup_block() {
    if (m_input_block->size() == 0) {
        m_input_block->resize(m_label_block->size());
    }
    CHECK(m_label_block->dims().size() == 2);
    CHECK(m_label_block->size() > 0);
    CHECK(m_input_block->size() == m_label_block->dims()[1]);
};

void softmax_loss_layer::setup_params() {
    this->m_loss = 0;
}

bool softmax_loss_layer::begin_seq() {
    this->m_output_history.clear();
    return true;
}

bool softmax_loss_layer::forward(int t){
    auto& input = m_input_block->signal();
    array2d label = m_label_block->signal();
    array output = m_input_block->signal().clone();
    m_output_history.push_back(output);

    //find max val
    float mmax = input.max();

    //exp(val-max)
    #pragma omp parallel for
    for (int i = 0; i < input.size(); ++i) {
        float dv = input.at(i) - mmax;
        output.at(i) = exp(dv);
    }


    //normalize
    float sum = output.sum();
    output.mul(1.0f / sum);

    //compute loss
    #pragma omp parallel for
    for (int i = 0; i < output.size(); ++i) {
        float lb = label.at2(t, i);
        if (fabs(lb) > 1e-5) {
            m_loss_sum -= lb * log(1e-15f + output.at(i));
        }
    }
    m_loss_num += 1;

    if (m_report) {
        for (int i = 0; i < output.size(); ++i) {
            cout << i
                << "\t" << input.at(i)
                << "\t" << output.at(i)
                << "\t" << label.at2(t, i) << endl;
        }
        cout << "==============" << endl << endl;
    }

    //stop
    return (t + 1) < label.rows();
}

void softmax_loss_layer::backward(int t) {
    auto& error = m_input_block->error();
    auto& output = m_output_history.back();
    array2d label = m_label_block->signal();
    m_output_history.pop_back();

    #pragma omp parallel for
    for (int i = 0; i < error.size(); ++i) {
        error.at(i) = label.at2(t, i) - output.at(i);
    }
}

void softmax_loss_layer::end_batch(int size) {
    m_loss = m_loss_sum / m_loss_num;
    m_loss_sum = 0;
    m_loss_num = 0;
}

layer_ptr create_softmax_loss_layer(
    const picojson::value& config,
    block_factory& bf){
    int input_id = (int)config.get("input").get<double>();
    int label_id = (int)config.get("label").get<double>();
    auto input_block = bf.get_block(input_id);
    auto label_block = bf.get_block(label_id);
    return layer_ptr(new softmax_loss_layer(input_block, label_block));
}

REGISTER_LAYER(softmax_loss, create_softmax_loss_layer);