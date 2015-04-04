#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "layer.h"

class gru_layer : public layer {
public:
    gru_layer(block_ptr input, block_ptr output, int output_num);

    virtual void setup_block();
    virtual void setup_params();
    virtual bool begin_seq();
    virtual bool forward();
    virtual void backward();
    virtual void end_batch(int size);

    virtual void save(std::ostream& os);
    virtual void load(std::istream& is);

    void create_layers();

private:
    std::vector<layer_ptr> m_layers;

    block_ptr m_input_block;
    block_ptr m_output_block;

    layer_ptr m_r_product_layer;
    block_ptr m_r_product_block;
    layer_ptr m_r_sigmoid_layer;
    block_ptr m_r_sigmoid_block;

    layer_ptr m_z_product_layer;
    block_ptr m_z_product_block;
    layer_ptr m_z_sigmoid_layer;
    block_ptr m_z_sigmoid_block;

    layer_ptr m_rhb_product_layer;
    block_ptr m_rhb_product_block;

    layer_ptr m_hb_product_layer;
    block_ptr m_hb_product_block;

    layer_ptr m_hb_tanh_layer;
    block_ptr m_hb_tanh_block;

    layer_ptr m_h_scale_layer;
    block_ptr m_h_scale_block;

    layer_ptr m_hb_wise_prod_layer;
    block_ptr m_hb_wise_prod_block;

    layer_ptr m_h_wise_prod_layer;
    block_ptr m_h_wise_prod_block;

    layer_ptr m_h_add_layer;

    int m_output_num;
};



#endif