#ifndef MULTI_SOFTMAX_LOSS_LAYER
#define MULTI_SOFTMAX_LOSS_LAYER

#include "layer.h"

class multi_softmax_loss_layer : public loss_layer {
public:
    multi_softmax_loss_layer(
        const std::vector<block_ptr> &input_blocks, 
        block_ptr &label_block);

    virtual void setup_block();
    virtual void setup_params();
    virtual bool begin_seq();
    virtual bool forward(int t);
    virtual void backward(int t);

    virtual void end_batch(int t);

    virtual float loss();

private:
    std::vector<block_ptr> m_input_blocks;
    std::vector<std::vector<array> > m_output_history;
    block_ptr m_label_block;
    float m_loss_sum;
    int m_loss_num;
};

#endif
