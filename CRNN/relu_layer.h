#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

class relu_layer : public layer {
public:
    relu_layer(
        std::shared_ptr<block> input_block,
        std::shared_ptr<block> output_block,
        float negtive_slope = -1.0f);
    virtual void setup_block();
    virtual bool forward(int t);
    virtual void backward(int t);
    virtual bool begin_seq();

    virtual void save(std::ostream& os);
    virtual void load(std::istream& is);

private:
    std::shared_ptr<block> m_input_block;
    std::shared_ptr<block> m_output_block;
    std::vector<array> m_input_history;
    float m_negtive_slop;

    inline float relu(float x) {
        return x >= 0.0f ? x : x * m_negtive_slop;
    }

    inline float rrelu(float x) {
        return x >= 0 ? 1.0f : m_negtive_slop;
    }
};

#endif