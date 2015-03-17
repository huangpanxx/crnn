#include "layer.h"
using namespace std;

layer::layer() { 
    m_learn_rate = 0.005f + (::rand() / RAND_MAX)*0.005f;
    m_momentum_decay = 0.9f;
    m_name = "";
    m_enable_bp = true;
}

std::map<string, layer_factory_fn>& get_layer_fns(){
    static std::map<string, layer_factory_fn> fns;
    return fns;
}

void register_layer_factory(string name, layer_factory_fn fn){
    get_layer_fns()[name] = fn;
}

layer_factory_fn get_layer_factory(string name){
    auto fns = get_layer_fns();
    if (fns.count(name) == 0) {
        printf("layer type :%s is unknown.\n", name.c_str());
        CHECK(false);
    }
    return fns[name];
}
