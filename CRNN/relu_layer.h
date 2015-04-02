#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

class relu_layer : public layer {
public:
    relu_layer(
        std::shared_ptr<block> input_block,
        std::shared_ptr<block> output_block,
        bool share = false);

    virtual void setup_block();
    virtual void setup_params();

    virtual bool forward(int t);
    virtual void backward(int t);
    virtual bool begin_seq();

    virtual void save(std::ostream& os);
    virtual void load(std::istream& is);

    virtual void end_batch(int size);

private:
    std::shared_ptr<block> m_input_block;
    std::shared_ptr<block> m_output_block;
    std::vector<arraykd> m_input_history;
    bool m_share;

    arraykd m_negtive_slop;
    arraykd m_negtive_slop_grad;
};

#endif