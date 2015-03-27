#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "layer.h"

class gru_layer : public layer {
public:
    gru_layer(block_ptr input, block_ptr output, int output_num, int hide_num);

    virtual void setup_block();
    virtual void setup_params();
    virtual bool begin_seq();
    virtual bool forward(int t);
    virtual void backward(int t);
    virtual void end_batch(int size);

private:
    block_ptr m_hide_block;
    block_ptr m_input_block;
    block_ptr m_output_block;
    int m_output_num;
    int m_hide_num;
};



#endif