#include "array_layer.h"
#include "utility.h"
using namespace std;

array_layer::array_layer(
    std::vector<array_sample> samples,
    std::shared_ptr<block> data,
    std::shared_ptr<block> label,
    int batch,int iter,int loop) {
    CHECK(samples.size() != 0);
    cout << samples.size() << " samples loaded ..." << endl;
    this->m_label = label;
    this->m_data = data;
    this->m_samples = samples;
    this->m_batch = batch;
    this->m_loop = loop;
    this->m_iter = iter;
    this->m_index = -1;
    auto data_dims = samples[0].data().dims();
    auto label_dims = samples[0].label().dims();
    for (auto&sample : samples) {
        CHECK(cmp_vec(data_dims, sample.data().dims()));
        CHECK(cmp_vec(label_dims, sample.label().dims()));
    }

    cout << "data dims:";
    for (int i = 0; i < (int)data_dims.size(); ++i){
        if (i) cout << ",";
        cout << data_dims[i];
    }
    cout << endl;
}

bool array_layer::begin_seq() {
   int max_index = (int)this->m_samples.size() * this->m_iter * m_loop;
   CHECK(max_index > 0);
   bool ok = m_index + 1 < max_index;
   if (ok) {
       m_index += 1;
   }
   this->m_data->error().clear(0);
   this->m_label->error().clear(0);
   return ok;
}



void array_layer::setup_block() {
    auto& sample = this->m_samples[0];
    if (this->m_data->empty()) {
        this->m_data->resize(sample.data().dims());
        this->m_label->resize(sample.label().dims());
    } else {
        CHECK(cmp_vec(this->m_data->dims(),sample.data().dims()));
        CHECK(cmp_vec(this->m_label->dims(), sample.label().dims()));
    }
}

bool array_layer::forward(int t) {
    //write data & label
    if (t != 0) {
        return false;
    }
    int group = m_iter*m_batch;
    int k = m_index / group;
    int idx = (k * m_batch + (m_index % group) % m_batch) % (int) m_samples.size();

    auto &sample = this->m_samples[idx];
    this->m_data->new_signal() = sample.data();
    this->m_label->new_signal() = sample.label();

    return true;
}

void array_layer::move_to_next_batch() {
    //move
    int group = m_iter * m_batch;
    m_index += group - (m_index % group) - 1;

    //print
    int k = m_index / group;
    int idx = (k * m_batch + (m_index % group) % m_batch) % (int) m_samples.size();
    printf("data continued from index %d(%d).\n", idx, m_index);
}

void array_layer::save(std::ostream& os) {
    write_val_to_stream(os, (int)m_index);
}

void array_layer::load(std::istream& is) {
    read_val_from_stream(is, m_index);
    if (m_index >= (int)this->m_samples.size() * this->m_iter * m_loop 
        || m_index < 0) {
        m_index = 0;
    }
    this->move_to_next_batch();
}
