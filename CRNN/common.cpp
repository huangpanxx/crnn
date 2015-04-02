#include "common.h"

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

using namespace std;

//this function will never be called
//it exists just force linker to link all the layer files
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
    return true;
}

//make a reference to dummy_fn
static bool initializer() {
    printf("crnn initialized.\n");
    return  dummy_fn != NULL;
}

bool CRNN_INITIALIZER = initializer();
