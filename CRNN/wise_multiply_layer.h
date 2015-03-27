#ifndef WISE_MULTIPLY_LAYER_H
#define WISE_MULTIPLY_LAYER_H

#include "layer.h"

class wise_multiply_layer : public layer {
public:
    wise_multiply_layer(block_ptr input1, block_ptr input2, block_ptr output);

    virtual void setup_block();
    virtual bool begin_seq();
    virtual bool forward(int t);
    virtual void backward(int t);

private:
    block_ptr m_input_block1;
    block_ptr m_input_block2;
    block_ptr m_output_block;
    std::vector<std::pair<array, array> > m_input_history;
};

#endif