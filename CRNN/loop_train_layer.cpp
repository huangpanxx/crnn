#include "loop_train_layer.h"

loop_train_layer::loop_train_layer(const std::vector<layer_ptr>& layers){
    this->m_layers = layers;
}

void loop_train_layer::setup_block(){
    for (auto &layer : m_layers){
        layer->setup_block();
    }
}

void loop_train_layer::setup_params(){
    for (auto &layer : m_layers){
        layer->setup_params();
    }
}

bool loop_train_layer::begin_seq(){
    this->m_t = 0;
    this->m_history.clear();
    for (auto& layer : m_layers){
        if (!layer->begin_seq()){
            return false;
        }
    }
    return true;
}

bool loop_train_layer::forward(int t){
    while (true){
        for (auto& layer : m_layers){
            m_history.push_back({ layer, m_t });
            if (!layer->forward(m_t)){
                goto end;
            }
        }
        ++m_t;
    }
end:
    return true;
}

void loop_train_layer::backward(int t){
    for (auto &pair : m_history){
        auto& layer = pair.first;
        auto& t = pair.second;
        layer->backward(t);
    }
}

void loop_train_layer::end_batch(int size){
    for (auto& layer : m_layers){
        layer->end_batch(size);
    }
}
