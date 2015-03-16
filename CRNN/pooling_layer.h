#ifndef POOLING_LAYER
#define POOLING_LAYER

#include "layer.h"

class max_pooling_layer : public layer {
public:
    max_pooling_layer(block_ptr input_block, block_ptr output_block, int size);

    virtual void setup_block();

    virtual bool begin_seq();
    virtual bool forward(int t);
    virtual void backward(int t);

private:
    block_ptr m_input_block;
    block_ptr m_output_block;
    std::vector<array3d> m_max_history;
    int m_size;
};

//class mean_pooling_layer : public layer{
//public:
//    mean_pooling_layer(block_ptr input_block, block_ptr output_block);
//};

#endif