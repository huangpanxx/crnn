#ifndef ARRAY_LAYER_H
#define ARRAY_LAYER_H

#include "memory.h"
#include "layer.h"

class array_sample {
public:
    array_sample(const array& data, const array& label){
        this->m_data = data;
        this->m_label = label;
    }
    array& data() { return m_data; }
    array& label() { return m_label; }
private:
    array m_data;
    array m_label;
};

//samples must be sure have the same dims(no check in this layer)
class array_layer : public data_layer {
public:
    array_layer(
        std::vector<array_sample> samples,
        std::shared_ptr<block> data,
        std::shared_ptr<block> label,
        int batch, int iter,int loop);

    virtual void setup_block();

    virtual bool begin_seq();
    virtual bool forward(int t);

    virtual int batch() { return m_batch; }
    int iter() { return m_iter; }
    int loop() { return m_loop; }

    std::vector<int> label_dims() {
        return this->m_samples[0].label().dims();
    }

    std::vector<int> data_dims() {
        return this->m_samples[0].data().dims();
    }

    int sample_num() {
        return (int)m_samples.size();
    }

    void set_index(int index) {
        this->m_index = index;
    }

    int index() {
        return m_index;
    }

   virtual void move_to_next_batch();

   virtual void save(std::ostream& os);
   virtual void load(std::istream& is);

private:
    std::shared_ptr<block> m_data;
    std::shared_ptr<block> m_label;
    std::vector<array_sample> m_samples;
    int m_index;
    int m_batch;
    int m_iter;
    int m_loop;
};

#endif