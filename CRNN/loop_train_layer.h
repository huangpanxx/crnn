#ifndef LOOP_TRAIN_LAYER_H
#define LOOP_TRAIN_LAYER_H

#include "layer.h"

class loop_train_layer : public layer{
public:
    loop_train_layer(const std::vector<layer_ptr>& layers);

    bool begin_seq();
    bool forward();
    void backward();

private:
    std::vector<layer_ptr> m_layers;
    std::vector<layer_ptr> m_forward_history;
};

#endif