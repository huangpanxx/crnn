#include "network.h"
#include "utility.h"
#include "loop_train_layer.h"
using namespace std;

network::network(const std::string& config, const std::string& plan) {
    //parse to json
    string err = picojson::parse(m_config, config);
    if (!err.empty()) {
        cerr << err << endl;
        CHECK(err.empty());
    }

    //learn rate
    CHECK(m_config.contains("learn_rate"));
    m_learn_rate = (float) m_config.get("learn_rate").get<double>();
    CHECK(m_learn_rate >= 0);

    //input dims
    CHECK(m_config.contains("input_dims"));
    auto dims = m_config.get("input_dims").get<picojson::array>();
    for (auto jdim : dims) {
        int d = (int) jdim.get<double>();
        m_input_dims.push_back(d);
    }

    //dict
    CHECK(m_config.contains("dict"));
    auto dict = m_config.get("dict").get<picojson::array>();
    for (int i = 0; i < (int) dict.size(); ++i) {
        auto val = dict[i].get<string>();
        m_label_dict[i] = val;
    }

    CHECK(m_config.contains(plan));
    auto plan_config = m_config.get(plan);

    //predict
    if (plan_config.contains("input")) {
        CHECK(plan_config.contains("output"));
        m_input_block_id = plan_config.get("input").get<string>();
        m_output_block_id = plan_config.get("output").get<string>();
        auto input_block = m_block_factory.get_block(m_input_block_id);
        input_block->resize(m_input_dims);
    }
    else{
        m_input_block_id = "";
        m_output_block_id = "";
    }

    //create layers
    CHECK(plan_config.contains("setup_block"));
    auto setup_block_seq = get_layers(plan_config.get("setup_block"));

    auto setup_params_seq = setup_block_seq;
    if (plan_config.contains("setup_params")){
        auto setup_params_seq = get_layers(plan_config.get("setup_params"));
    }

    this->m_beg_layer_seq = setup_block_seq;
    if (plan_config.contains("begin_seq")){
        this->m_beg_layer_seq = get_layers(plan_config.get("begin_seq"));
    }

    if (plan_config.contains("activation")) {
        auto arr = plan_config.get("activation").get<picojson::array>();
        for (auto &cfg : arr){
            m_activate_layer_seq.push_back(get_layers(cfg));
        }
    }
    else{
        this->m_activate_layer_seq.push_back(setup_block_seq);
    }

    //train(loss data)
    if (plan_config.contains("loss")) {
        CHECK(plan_config.contains("data"));
        auto data_layer_name = plan_config.get("data").get<string>();
        auto loss_layer_name = plan_config.get("loss").get<string>();
        m_data_layer = (data_layer*) get_layer(data_layer_name).get();
        m_loss_layer = (loss_layer*) get_layer(loss_layer_name).get();
    }
    else{
        m_data_layer = 0;
        m_loss_layer = 0;
    }

    //model file
    m_model_file = m_config.get("model").get<string>();
    ifstream is(m_model_file, ios::binary | ios::in);
    if (is) {
        load_layers(is, m_layer_cache);
        is.close();
    }

    //setup block
    for_each(setup_block_seq.begin(), setup_block_seq.end(),
        [](layer_ptr &x){ x->setup_block(); });

    //setup params
    for_each(setup_params_seq.begin(), setup_params_seq.end(),
        [](layer_ptr &x){ x->setup_params(); });

    //stop loss
    if (plan_config.contains("stop_loss")) {
        m_stop_loss = (float) plan_config.get("stop_loss").get<double>();
    }
    else{
        m_stop_loss = 0.0f;
    }


    //save epoch
    this->m_save_epoch = (int) m_config.get("save_epoch").get<double>();
    CHECK(m_save_epoch >= 1);
}


vector<layer_ptr> network::get_layers(const picojson::value& val){
    auto arr = val.get<picojson::array>();
    vector<layer_ptr> layers;
    for (auto val : arr){
        if (val.is<string>()){
            auto layer = get_layer(val.get<string>());
            layers.push_back(layer);
        }
        if (val.is<picojson::array>()) {
            auto sub_layers = get_layers(val);
            layer_ptr loop_layer(new loop_train_layer(sub_layers));
            layers.push_back(loop_layer);
        }
    }
    return layers;
}

layer_ptr network::get_layer(const std::string &name) {
    if (m_layer_cache.count(name) == 0) {
        auto arr = m_config.get("layers").get<picojson::array>();
        bool is_created = false;
        for (auto v : arr) {
            if (v.get("name").get<string>() == name){
                auto type = v.get("type").get<string>();
                printf("create layer, type = %s, name = %s.\n",
                    type.c_str(), name.c_str());
                auto fn = get_layer_factory(type);
                auto layer = fn(v, name, this->m_block_factory);
                config_layer(v, name, layer);
                m_layer_cache[name] = layer;
                is_created = true;
                break;
            }
        }
        if (!is_created){
            printf("%s is not registed.\n", name.c_str());
            CHECK(is_created);
        }
    }
    return m_layer_cache[name];
}

vector<layer_ptr> network::get_layers(const std::vector<string> &names) {
    vector<layer_ptr> layer_seq;
    for (auto name : names){
        auto layer = get_layer(name);
        layer_seq.push_back(layer);
    }
    return layer_seq;
}

void network::config_layer(picojson::value& val, const string& layer_name, layer_ptr& layer){
    layer->set_name(layer_name);
    layer->set_learn_rate(m_learn_rate);

    if (val.contains("learn_rate")) {
        float lr = (float) val.get("learn_rate").get<double>();
        layer->set_learn_rate(lr);
    }

    if (val.contains("momentum")) {
        float momentum = (float) val.get("momentum").get<double>();
        layer->set_momentum_decay(momentum);
    }

    if (val.contains("enable_bp")){
        bool b = val.get("enable_bp").get<bool>();
        layer->set_enable_bp(b);
        if (!b){
            printf("layer %s disable bp.\n", layer->name().c_str());
        }
    }

    printf("layer %s, learn_rate = %.8f, momentum = %0.8f.\n",
        layer_name.c_str(),
        layer->learn_rate(),
        layer->momentum_decay());
}


void network::train() {
    CHECK(this->m_data_layer);
    CHECK(this->m_loss_layer);

    const int batch = m_data_layer->batch();
    CHECK(batch >= 1);

    auto start_time = clock();
    for (int iter = 0; ; ++iter) {
        const int epoch = iter / batch;
        //batch
        if (iter % batch == 0 && iter) {
            //print info
            float freq = (float) (batch) * CLOCKS_PER_SEC / (clock() - start_time);
            float loss = m_loss_layer->loss();
            printf("epoch %d, %.3f iter/s, loss = %.8f.                     \n",
                epoch, freq, loss);

            //end batch
            for (auto &layer : this->m_beg_layer_seq) {
                layer->end_batch(batch);
            }

            //save
            if (epoch % m_save_epoch == m_save_epoch - 1) {
                printf("save model ...\n");
                ofstream os(m_model_file, ios::binary | ios::out);
                CHECK(os);
                save_layers(os, m_layer_cache);
                os.close();
            }

            //skip
            if (loss < m_stop_loss) {
                printf("move to next batch.\n");
                m_data_layer->move_to_next_batch();
            }
            start_time = clock();
        }

        //begin
        for (auto &layer : this->m_beg_layer_seq) {
            if (!layer->begin_seq()) {
                goto finish_train;
            }
        }

        vector<layer_ptr> forward_history;

        //forward
        for (auto &layers : this->m_activate_layer_seq){
            for (auto& layer : layers){
                forward_history.push_back(layer);
                if (!layer->forward()){
                    goto end_forward;
                }
            }
        }
        end_forward:

        //backward
        for_each(forward_history.rbegin(), 
            forward_history.rend(),
            [](layer_ptr &layer) {
                layer->backward();
        });

        //info
        printf("epoch %d, iter %d.\r", epoch, iter);
    }
    finish_train:
    printf("train finished.\n");
}

void network::set_input(const arraykd& data){
    CHECK(m_input_block_id != "");
    this->m_t = 0;
    auto &input = this->m_block_factory.get_block(this->m_input_block_id);
    input->signal() = data; //we don't need input has the same size
}

arraykd network::forward(){
    //beg
    if (m_t == 0) {
        for (auto &layer : m_beg_layer_seq){
            layer->begin_seq();
        }
    }

    //forward
    int index = std::min((int)this->m_activate_layer_seq.size() - 1, m_t);
    auto acti_seq = this->m_activate_layer_seq[index];
    for (auto &layer : acti_seq){
        layer->forward();
    }

    m_t += 1;
    return m_block_factory.get_block(m_output_block_id)->signal();
}
