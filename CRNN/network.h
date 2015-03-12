#ifndef NETWORK_H
#define NETWORK_H


#include "layer.h"

class network {
public:
    network(const std::string& config, const std::string& plan);

    block_ptr block(int id) {
        CHECK(m_block_factory.contains(id));
        return m_block_factory.get_block(id);
    }

    //train
    void train();

    //predict
    void set_input(const array& data);
    array forward();

    //dict
    const std::string& translate(int k) {
        CHECK(m_label_dict.count(k) != 0);
        return m_label_dict[k];
    }

private:
    layer_ptr get_layer(const std::string &name);
    std::vector<layer_ptr> get_layers(const picojson::value& val);
    std::vector<layer_ptr> get_layers(const std::vector<std::string> &names);
    void config_layer(picojson::value& val, layer_ptr& layer);
    std::map<int, std::string> m_label_dict;

private:
    block_factory m_block_factory;
    std::vector<std::vector<layer_ptr> > m_activate_layer_seq;
    std::vector<layer_ptr> m_beg_layer_seq;
    std::map<std::string, layer_ptr> m_layer_cache;
    std::shared_ptr<data_layer> m_data_layer;
    std::shared_ptr<loss_layer> m_loss_layer;
    std::string m_model_file;
    picojson::value m_config;
    std::vector<int> m_input_dims;
    int m_save_epoch;
    int m_input_block_id;
    int m_output_block_id;
    int m_t;
    float m_stop_loss;
    float m_learn_rate;
};

#endif