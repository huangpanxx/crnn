#ifndef MULTI_SOFTMAX_LAYER
#define MULTI_SOFTMAX_LAYER

#include "layer.h"

class multi_softmax_layer : public layer{
public:
    multi_softmax_layer(
        const std::vector<block_ptr> &input_blocks,
        block_ptr &output_block);

    virtual void setup_block();
    virtual bool forward(int t);
    virtual void backward(int t);
    virtual void end_batch(int t);

private:
    std::vector<block_ptr> m_input_blocks;
    block_ptr m_output_block;
};

#endif