#ifndef SOFTMAX_LOSS_LAYER_H
#define SOFTMAX_LOSS_LAYER_H

#include "layer.h"
#include "memory.h"

class softmax_loss_layer : public loss_layer {
public:
    softmax_loss_layer(
        std::shared_ptr<block> input_block,
        std::shared_ptr<block> label_block);

    virtual void setup_block();
    virtual bool begin_seq();
    virtual bool forward(int t);
    virtual void backward(int t);

    virtual void end_batch(int size);

    void set_report(bool b){
        m_report = b;
    }

    virtual float loss(){ 
        float loss = 0;
        if (m_loss_num != 0){
            loss = m_loss_sum / m_loss_num;
        }
        return loss;
    }

private:
    std::shared_ptr<block> m_input_block;
    std::shared_ptr<block> m_label_block;
    std::vector<array> m_output_history;
    float m_loss_sum;
    int m_loss_num;
    bool m_report;
};


#endif