#include "image_data_layer.h"
#include "utility.h"
#include <fstream>
#include <sstream>

using namespace std;

vector<array_sample> load_images(const string& dirname, const string& label_file){
    ifstream fin(label_file);
    CHECK(fin);
    int label_size;
    fin >> label_size;
    CHECK(label_size > 0);
    vector<array_sample> samples;
    while (fin) {
        string line;
        while (fin) {
            getline(fin, line);
            if (!line.empty())
                break;
        }
        int pos = line.find(' ');
        if (pos <= 0) continue;
        string file_name = line.substr(0, pos);
        string label_str = line.substr(pos);
        CHECK(!file_name.empty());
        stringstream ss(label_str);
        int label_num;
        ss >> label_num;
        vector<int> vlabel;
        for (int i = 0; i<label_num; ++i){
            int nlabel;
            ss >> nlabel;
            vlabel.push_back(nlabel);
        }
        string file_path = dirname + "/" + file_name;
        array3d image = imread(file_path);
        if (image.size() == 0){
            cout << "load image " << file_name << "failed" << endl;
            continue;
        }
        array2d label((int)vlabel.size(),label_size);
        label.clear(0);
        for (int i = 0; i<(int) vlabel.size(); ++i){
            label.at2(i, vlabel[i]) = 1.0f;
        }
        samples.push_back(array_sample(image, label));
        if (samples.size() > 2) {
            CHECK(cmp_array_dim(samples.front().data(), samples.back().data()));
        }
    }
    CHECK(samples.size() > 0);
    return samples;
}

image_data_layer::image_data_layer(
    const string& dirname,
    const string& label_file,
    shared_ptr<block> data,
    shared_ptr<block> label,
    int batch, int iter,int loop)
    :array_layer(load_images(dirname, label_file), data, label, batch, iter,loop) {
}


layer_ptr create_image_layer(
    const picojson::value& config,
    block_factory& bf) {
    string label_file = config.get("label_file").get<string>();
    string data_dir = config.get("data_dir").get<string>();
    int data_id = (int) config.get("data").get<double>();
    int label_id = (int) config.get("label").get<double>();
    int batch = (int) config.get("batch").get<double>();
    int iter = (int) config.get("iter").get<double>();
    int loop = (int) config.get("loop").get<double>();

    auto data_block = bf.get_block(data_id);
    auto label_block = bf.get_block(label_id);

    return layer_ptr(
        new image_data_layer(data_dir, label_file,
        data_block, label_block, batch, iter, loop));
}

REGISTER_LAYER(image_data, create_image_layer);