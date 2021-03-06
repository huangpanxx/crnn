#ifndef ADD_LAYER_H
#define ADD_LAYER_H

#include "layer.h"


class add_layer : public layer{
public:
    add_layer(const std::vector<block_ptr> &input_blocks, block_ptr &output_block);
    virtual void setup_block();
    virtual bool forward();
    virtual void backward();
    virtual bool begin_seq();

private:
    std::vector<block_ptr> m_input_blocks;
    block_ptr m_output_block;
};

#endif