#include "gru_layer.h"
#include "inner_product_layer.h"
#include "sigmoid_layer.h"
#include "wise_product_layer.h"
#include "tanh_layer.h"
#include "scale_layer.h"
#include "add_layer.h"
#include "network.h"
using namespace std;

gru_layer::gru_layer(block_ptr input, block_ptr output, int output_num) {
    this->m_input_block = input;
    this->m_output_block = output;
    this->m_output_num = output_num;
    create_layers();
}

void gru_layer::setup_block() {
    if (this->m_output_block->empty()){
        this->m_output_block->resize(m_output_num);
    }
    CHECK(this->m_output_block->size() == m_output_num);

    for (auto& layer : m_layers){
        layer->setup_block();
    }
}

void gru_layer::create_layers() {
    this->m_hb_product_block = block::new_block();
    this->m_hb_tanh_block = block::new_block();
    this->m_hb_wise_prod_block = block::new_block();
    this->m_h_scale_block = block::new_block();
    this->m_h_wise_prod_block = block::new_block();
    this->m_rhb_product_block = block::new_block();
    this->m_r_product_block = block::new_block();
    this->m_r_sigmoid_block = block::new_block();
    this->m_z_product_block = block::new_block();
    this->m_z_sigmoid_block = block::new_block();

    //r
    this->m_r_product_layer.reset(new inner_product_layer(
    { m_input_block, m_output_block },
    m_r_product_block, m_output_num));
    m_layers.push_back(m_r_product_layer);

    this->m_r_sigmoid_layer.reset(new sigmoid_layer(
        m_r_product_block,
        m_r_sigmoid_block));
    m_layers.push_back(m_r_sigmoid_layer);

    //z
    this->m_z_product_layer.reset(new inner_product_layer(
    { m_input_block, m_output_block },
    m_z_product_block, m_output_num));
    m_layers.push_back(m_z_product_layer);

    this->m_z_sigmoid_layer.reset(new sigmoid_layer(
        m_z_product_block,
        m_z_sigmoid_block));
    m_layers.push_back(m_z_sigmoid_layer);

    //rhb
    this->m_rhb_product_layer.reset(new wise_product_layer(
        m_r_product_block, m_output_block, m_rhb_product_block));
    m_layers.push_back(m_rhb_product_layer);

    //hb
    this->m_hb_product_layer.reset(new inner_product_layer(
    { m_input_block, m_rhb_product_block },
    m_hb_product_block, m_output_num));
    m_layers.push_back(m_hb_product_layer);

    //hb_tanh
    this->m_hb_tanh_layer.reset(new tanh_layer(
        m_hb_product_block, m_hb_tanh_block));
    m_layers.push_back(m_hb_tanh_layer);

    //h_scale
    this->m_h_scale_layer.reset(new scale_layer(
        m_z_sigmoid_block, m_h_scale_block, -1.0, 1));
    m_layers.push_back(m_h_scale_layer);

    //h_wise_product
    this->m_h_wise_prod_layer.reset(new wise_product_layer(
        m_h_scale_block, m_output_block, m_h_wise_prod_block));
    m_layers.push_back(m_h_wise_prod_layer);

    //hb_wise_product
    this->m_hb_wise_prod_layer.reset(new wise_product_layer(
        this->m_z_sigmoid_block, this->m_hb_tanh_block,
        m_hb_wise_prod_block));
    m_layers.push_back(m_hb_wise_prod_layer);

    //add
    this->m_h_add_layer.reset(new add_layer(
    { m_hb_wise_prod_block, m_h_wise_prod_block }, this->m_output_block));
    m_layers.push_back(m_h_add_layer);
}

void gru_layer::setup_params() {
    for (auto& layer : m_layers){
        layer->setup_params();
    }
}

bool gru_layer::begin_seq(){
    for (auto& layer : m_layers){
        layer->begin_seq();
    }

    return true;
}

bool gru_layer::forward() {
    for (auto& layer : m_layers){
        layer->forward();
    }
    return true;
}

void gru_layer::backward() {
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it){
        (*it)->backward();
    }

}

void gru_layer::end_batch(int size) {
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it){
        (*it)->end_batch(size);
    }
}

void gru_layer::save(std::ostream& os){
    for (auto& layer : m_layers){
        layer->save(os);
    }
}

void gru_layer::load(std::istream& is){
    for (auto& layer : m_layers){
        layer->load(is);
    }
}

layer_ptr create_gru_layer(
    const picojson::value& config,
    const string& layer_name,
    network* net) {
    CHECK(config.contains("input"));
    CHECK(config.contains("output_num"));

    string input_block_id =  config.get("input").get<string>();
    int output_num = (int)config.get("output_num").get<double>();
    auto input_block = net->block(input_block_id);
    auto output_block = net->block(layer_name);
    return layer_ptr(new gru_layer(input_block, output_block, output_num));
}

REGISTER_LAYER(gru, create_gru_layer);