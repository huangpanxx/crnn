#include "layer.h"
#include "image_split_layer.h"
using namespace std;

layer::layer() {
    m_learn_rate = 0.005f + (::rand() / RAND_MAX)*0.005f;
    m_momentum_decay = 0.9f;
    m_name = "unname";
    m_enable_bp = true;
    m_counter = 0;
    this->m_array_operator = get_default_array_operator();
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



//=========== linker helper ==========

#include "conv_layer.h"
#include "sigmoid_layer.h"
#include "relu_layer.h"
#include "multi_softmax_layer.h"
#include "inner_product_layer.h"
#include "gru_layer.h"
#include "softmax_loss_layer.h"
#include "add_layer.h"
#include "wise_product_layer.h"
#include "tanh_layer.h"
#include "pooling_layer.h"
#include "multi_softmax_loss_layer.h"
#include "image_data_layer.h"
#include "softmax_layer.h"

//this function will never be called
//it just forces linker to link all the layer files
static bool dummy_fn(){
    conv_layer(0, 0, 2, 2, 2);
    relu_layer(0, 0, false);
    sigmoid_layer(0, 0);
    multi_softmax_layer({}, block_ptr(0));
    inner_product_layer(0, 0, 1);
    gru_layer(0, 0, 1);
    softmax_loss_layer(0, 0);
    add_layer({}, block_ptr(0));
    wise_product_layer(0, 0, 0);
    tanh_layer(0, 0);
    max_pooling_layer(0, 0, 1, 1);
    multi_softmax_loss_layer({}, block_ptr(0));
    image_data_layer("", "", 0, 0, 1, 1, 1, 1);
    softmax_layer(0, 0);
    image_split_layer("", "", 0, 0, 0, 0, 0, 0, 0);
    image_slice_layer(0, 0, 0, 0);
    return true;
}

//make a reference to dummy_fn
static bool initializer() {
    //printf("crnn initialized.\n");
    return  dummy_fn != NULL;
}

static bool CRNN_INITIALIZER = initializer();
