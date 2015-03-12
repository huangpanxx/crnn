#ifndef LAYER_H
#define LAYER_H

#include "common.h"
#include "memory.h"
#include "picojson.h"

class layer {
public:
    layer();
    virtual void setup_block(){};
    virtual void setup_params(){};
    virtual bool begin_seq(){ return true; }
    virtual bool forward(int t){ return true; };
    virtual void backward(int t){};
    virtual void end_batch(int size){};

    float learn_rate(){ return m_learn_rate; }
    float momentum_decay(){ return m_momentum_decay; }
    void set_learn_rate(float lr){ this->m_learn_rate = lr; }
    void set_momentum_decay(float md) { this->m_momentum_decay = md; }

    virtual void save(std::ostream& os) { }
    virtual void load(std::istream& is) { }

    std::string name(){ return m_name; }
    void set_name(const std::string name){ m_name = name; }

private:
    float m_learn_rate;
    float m_momentum_decay;
    std::string m_name;
};

class loss_layer: public layer {
public:
    virtual float loss() = 0;
};

class data_layer : public layer {
public:
    virtual int batch() = 0;
    virtual void move_to_next_batch() = 0;
};


typedef std::shared_ptr<layer> layer_ptr;

typedef std::function<layer_ptr(const picojson::value& config,
    block_factory& bf)> layer_factory_fn;

void register_layer_factory(std::string name, layer_factory_fn fn);
layer_factory_fn get_layer_factory(std::string name);

#define REGISTER_LAYER(name,fn) \
    bool name##_layer_create_fn() { \
    register_layer_factory(#name, fn); \
    return 1; \
    }\
    bool name##_layer_create_var = name##_layer_create_fn();

#endif
