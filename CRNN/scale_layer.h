#ifndef SCALE_LAYER_H
#define SCALE_LAYER_H


#include "layer.h"

class scale_layer : public layer {
public:
    scale_layer(block_ptr input, block_ptr output, float scale, float bias);

    virtual void setup_block();
    virtual bool begin_seq();
    virtual bool forward();
    virtual void backward();

private:
    block_ptr m_input_block;
    block_ptr m_output_block;
    float m_scale;
    float m_bias;
};

#endif
