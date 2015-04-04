#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "layer.h"

class sigmoid_layer : public layer {
public:
    sigmoid_layer(
        std::shared_ptr<block> input_block,
        std::shared_ptr<block> output_block);
    virtual void setup_block();
    virtual bool forward();
    virtual void backward();
    virtual bool begin_seq();

private:
    std::shared_ptr<block> m_input_block;
    std::shared_ptr<block> m_output_block;
    std::vector<arraykd> m_output_history;

    float sigmoid(float x) {
        if (x > 20) return  0.999999f;
        if (x < -20) return 0.000001f;
        return 1.0f / (1.0f + exp(-x));
    }
};

#endif