#ifndef CONV_LAYER
#define CONV_LAYER

#include "layer.h"

class conv_layer : public layer {
public:
    conv_layer(
        block_ptr input_block,
        block_ptr output_block,
        int kernel_size, 
        int kernel_num,
        int kernel_stride);

    virtual void setup_block();
    virtual void setup_params();
    virtual bool begin_seq();
    virtual bool forward(int t);
    virtual void backward(int t);
    virtual void end_batch(int size);

    std::vector<array3d> weights(){
        return m_weights;
    }

    void set_weights(const std::vector<array3d>& weights) {
        this->m_weights = weights;
    }

    array bias() {
        return m_bias;
    }

    void set_bias(const array& bias){
        this->m_bias = bias;
    }

    virtual void save(std::ostream& os);
    virtual void load(std::istream& is);

private:
    block_ptr m_input_block;
    block_ptr m_output_block;

    std::vector<array3d> m_weights;
    array m_bias;

    std::vector<array3d> m_grad_weights;
    array m_grad_bias;

    std::vector<array3d> m_input_history;

    int m_kernel_size;
    int m_kernel_num;
    int m_kernel_stride;
};

#endif