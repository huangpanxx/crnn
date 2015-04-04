#ifndef LOOP_TRAIN_LAYER_H
#define LOOP_TRAIN_LAYER_H

#include "layer.h"

class loop_train_layer : public layer{
public:
    loop_train_layer(const std::vector<layer_ptr>& layers);

    void setup_block();
    void setup_params();
    bool begin_seq();
    bool forward(int t);
    void backward(int t);
    void end_batch(int size);

private:
    int m_t;
    std::vector<layer_ptr> m_layers;
    std::vector<std::pair<layer_ptr,int> > m_history;
};

#endif