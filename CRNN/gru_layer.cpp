#include "gru_layer.h"
#include "inner_product_layer.h"
#include "sigmoid_layer.h"
#include "wise_product_layer.h"
#include "tanh_layer.h"
#include "scale_layer.h"
#include "add_layer.h"
using namespace std;

gru_layer::gru_layer(block_ptr input, block_ptr output, int output_num) {
    this->m_input_block = input;
    this->m_output_block = output;
    this->m_output_num = output_num;
}

void gru_layer::setup_block() {
    if (this->m_output_block->empty()){
        this->m_output_block->resize(m_output_num);
    }
    CHECK(this->m_output_block->size() == m_output_num);
}

void gru_layer::setup_params() {
    //r
    this->m_r_product_layer.reset(new inner_product_layer(
        m_output_num,
        { m_input_block, m_output_block },
        m_r_product_block));
    this->m_r_sigmoid_layer.reset(new sigmoid_layer(
        m_r_product_block,
        m_r_sigmoid_block));

    //z
    this->m_z_product_layer.reset(new inner_product_layer(
        m_output_num,
        { m_input_block, m_output_block },
        m_z_product_block));
    this->m_z_sigmoid_layer.reset(new sigmoid_layer(
        m_z_product_block,
        m_z_sigmoid_block));

    //rhb
    this->m_rhb_product_layer.reset(new wise_product_layer(
        m_r_product_block, m_output_block, m_rhb_product_block));

    //hb
    this->m_hb_product_layer.reset(new inner_product_layer(
        m_output_num, { m_input_block, m_rhb_product_block },
        m_hb_product_block));

    //hb_tanh
    this->m_hb_tanh_layer.reset(new tanh_layer(
        m_hb_product_block, m_hb_tanh_block));

    //h_scale
    this->m_h_scale_layer.reset(new scale_layer(
        m_z_sigmoid_block, m_h_scale_block, -1.0, 1));

    //h_wise_product
    this->m_h_wise_prod_layer.reset(new wise_product_layer(
        m_h_scale_block, m_output_block, m_h_wise_prod_block));

    //hb_wise_product
    this->m_hb_wise_prod_layer.reset(new wise_product_layer(
        this->m_z_sigmoid_block, this->m_hb_tanh_block,
        m_hb_wise_prod_block));

    //add
    this->m_h_add_layer.reset(new add_layer(
    { m_hb_wise_prod_block, m_h_wise_prod_block }, this->m_output_block));
}

bool gru_layer::begin_seq(){

}

bool gru_layer::forward(int t) {

}

void gru_layer::backward(int t) {

}

void gru_layer::end_batch(int size) {

}
