#ifndef SOFTMAX_LAYER
#define SOFTMAX_LAYER

#include "layer.h"
#include "memory.h"

class softmax_layer : public layer {
public:
    softmax_layer(
        std::shared_ptr<block> input_block,
        std::shared_ptr<block> output_block);

    virtual void setup_block();
    virtual bool forward(int t);
    virtual void backward(int t);
    virtual void end_batch(int size);

private:
    std::shared_ptr<block> m_input_block;
    std::shared_ptr<block> m_output_block;
};

#endif