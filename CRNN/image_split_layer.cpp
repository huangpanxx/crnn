#include "image_split_layer.h"
#include "utility.h"
#include "network.h"
using namespace std;

array3d resize(const array3d& src, int width, int height, int stride){
    int new_width = int(float(height) / src.rows() * src.cols() + 0.5f);
    new_width -= (new_width - width) % stride;
    if (new_width < width){
        new_width = width;
    }
    return resize(src, new_width, height);
}

image_slice_layer::image_slice_layer(block_ptr data_block,int width,int height,int stride){
    this->m_data_block = data_block;
    this->m_t = 0;
    this->m_width = width;
    this->m_height = height;
    this->m_stride = stride;
}

void image_slice_layer::setup_block(){
    this->m_data_block->resize(m_height,m_width,3);
}

bool image_slice_layer::begin_seq() {
    this->m_t = 0;
    return true;
}

bool image_slice_layer::forward() {
    if (m_t >= this->m_helper.image_slice_num()){
        return false;
    }
    auto slice = this->m_helper.image_slice(m_t);
    this->m_data_block->new_signal().copy(slice);
    ++m_t;
    return true;
}

void image_slice_layer::set_data(const arraykd& data){
    array3d image = data;
    CHECK(image.dim() == 3 && image.dim(2) == 3);
    this->m_helper = image_split_helper(image, m_width, m_height, m_stride);
    this->m_t = 0;
}

label_slice_layer::label_slice_layer(block_ptr label_block,int label_size){
    this->m_label_block = label_block;
    this->m_label_size = label_size;
}

void label_slice_layer::setup_block() {
    this->m_label_block->resize(m_label_size);
}

bool label_slice_layer::begin_seq(){
    this->m_t = 0;
    return true;
}

bool label_slice_layer::forward(){
    if (m_t + 1 >= (int)this->m_labels.size()){
        return false;
    }
    ++m_t;
    return true;
}

void label_slice_layer::backward(){
    int k = this->m_labels[m_t];
    auto& signal = this->m_label_block->new_signal();
    for (int i = 0; i < signal.size(); ++i){
        signal.at(i) = (k == i) ? 1.0f : 0.0f;
    }
    --m_t;
}

void label_slice_layer::set_label(const vector<int>& labels){
    this->m_labels = labels;
}


image_split_helper::image_split_helper(
    const array3d& image,
    int width, int height, int stride) : m_image(resize(image, width, height, stride)) {
    this->m_width = width;
    this->m_height = height;
    this->m_stride = stride;
    this->m_image = resize(image, width, height, stride);
}

array3d image_split_helper::image_slice(int k){
    array3d slice(m_height, m_width, 3);
    OMP_FOR
    for (int r = 0; r < m_height; ++r){
        for (int c = 0; c < m_width; ++c){
            for (int ch = 0; ch < 3; ++ch){
                slice.at3(r, c, ch) = m_image.at3(r, c + k*m_stride, ch);
            }
        }
    }
    return slice;
}

int image_split_helper::image_slice_num(){
    return (m_image.cols() - m_width) / m_stride + 1;
}


image_split_layer::image_split_layer(
    const std::string& label_file,
    const std::string& data_dir,
    int width, int height, int stride,
    int batch, int label_size,
    block_ptr data_block, block_ptr label_block) {
    this->m_width = width;
    this->m_height = height;
    this->m_stride = stride;
    this->m_batch = batch;
    this->m_index = 0;
    this->m_label_size = label_size;
    this->m_data_block = data_block;
    this->m_label_block = label_block;
    this->m_data_dir = data_dir;
    this->m_image_slice_layer.reset(new image_slice_layer(data_block, width, height, stride));
    this->m_label_slice_layer.reset(new label_slice_layer(label_block, label_size));
    this->m_samples = read_label_file(label_file);
    CHECK(!m_samples.empty());
}


void image_split_layer::setup_params(){
    this->m_image_slice_layer->setup_params();
    this->m_label_slice_layer->setup_params();
}

void image_split_layer::setup_block(){
    this->m_image_slice_layer->setup_block();
    this->m_label_slice_layer->setup_block();
}

bool image_split_layer::begin_seq(){
    this->m_image_slice_layer->begin_seq();
    this->m_label_slice_layer->begin_seq();
    int idx = (m_index++) % this->m_samples.size();
    auto &p_sample = m_samples[idx];
    auto filename = m_data_dir + "/" + p_sample.first;

    //TODO: USE LRU CACHE
    auto image = imread(filename);
    ((image_slice_layer*) this->m_image_slice_layer.get())->set_data(image);
    ((label_slice_layer*) this->m_label_slice_layer.get())->set_label(p_sample.second);
    return this->m_image_slice_layer->begin_seq() &&
        this->m_label_slice_layer->begin_seq();

}

bool image_split_layer::forward(){
    cout << "you should not call this function" << endl;
    CHECK(false);
    return true;
}

void image_split_layer::save(std::ostream& os){
    write_val_to_stream<int>(os, this->m_index);
}

void  image_split_layer::load(std::istream& is){
    this->m_index = read_val_from_stream<int>(is);
    cout << "image split data layer: index loaded. val = " << m_index << endl;
}

int image_split_layer:: batch(){
    return this->m_batch;
}

void image_split_layer::move_to_next_batch(){ }

static layer_ptr create_image_split_layer(
    const picojson::value& config,
    const string& layer_name,
    network* net) {
    CHECK(config.contains("label_file")); CHECK(config.contains("data_dir"));
    CHECK(config.contains("width")); CHECK(config.contains("height"));
    CHECK(config.contains("stride")); CHECK(config.contains("batch"));
    CHECK(config.contains("label_size"));
    auto label_file = config.get("label_file").get<string>();
    auto data_dir = config.get("data_dir").get<string>();
    auto width = (int) config.get("width").get<double>();
    auto height = (int) config.get("height").get<double>();
    auto stride = (int) config.get("stride").get<double>();
    auto batch = (int) config.get("batch").get<double>();
    auto label_size = (int) config.get("label_size").get<double>();
    auto data_block = net->block("data");
    auto label_block = net->block("label");
    auto layer = new image_split_layer(label_file, data_dir, width, height,
        stride, batch, label_size, data_block, label_block);
    layer->get_image_slice_layer()->set_name(layer_name + ".data");
    layer->get_label_slice_layer()->set_name(layer_name + ".label");
    net->add_layer(layer->get_image_slice_layer());
    net->add_layer(layer->get_label_slice_layer());
    return layer_ptr(layer);
}

REGISTER_LAYER(image_split_data, create_image_split_layer);

static layer_ptr create_image_slice_layer(
    const picojson::value& config,
    const string& layer_name,
    network* net) {
    CHECK(config.contains("width"));
    CHECK(config.contains("height"));
    CHECK(config.contains("stride"));
    int width = (int) config.get("width").get<double>();
    int height = (int) config.get("height").get<double>();
    int stride = (int) config.get("stride").get<double>();
    auto &block = net->block("data");
    return layer_ptr(new image_slice_layer(block, width, height, stride));
}

REGISTER_LAYER(image_slice, create_image_slice_layer);