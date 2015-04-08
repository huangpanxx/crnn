#include "loop_train_layer.h"
using namespace std;

loop_train_layer::loop_train_layer(const std::vector<layer_ptr>& layers){
    this->m_layers = layers;
}

bool loop_train_layer::begin_seq(){
    m_forward_history.clear();
    return true;
}

bool loop_train_layer::forward(){
    while (true){
        for (auto& layer : m_layers){
            m_forward_history.push_back(layer);
            if (!layer->forward_and_report()){
                goto forward_end;
            }
        }
    }
forward_end:
    return true;
}

void loop_train_layer::backward(){
    for_each(m_forward_history.rbegin(), m_forward_history.rend(),
        [](layer_ptr& layer){
        layer->backward_and_report();
    });
}

