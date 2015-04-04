#ifndef LOOP_TRAIN_LAYER_H
#define LOOP_TRAIN_LAYER_H

#include "layer.h"

class loop_train_layer : public layer{
public:
    loop_train_layer(const std::vector<layer_ptr>& layers);

    void setup_block();
    void setup_params();
    bool begin_seq();
    bool forward();
    void backward();
    void end_batch(int size);

private:
    std::vector<layer_ptr> m_layers;
    std::vector<layer_ptr> m_forward_history;
};

#endif