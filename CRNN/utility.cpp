#include "utility.h"
using namespace std;

extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
    void  stbi_image_free(void *retval_from_stbi_load);
}

array3d imread(const std::string& path){
    int w, h, n;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 3);
    CHECK(data);
    array3d image(h, w, n);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            unsigned char* pt = data + n * (y * w + x);
            for (int ch = 0; ch < n; ++ch) {
                float val = pt[ch] / 255.0f;
                image.at3(ch, y, x) = val;
            }
        }
    }
    stbi_image_free(data);
    return image;
}

const int BEGIN_MAGIC_NUMBER = 410927;
void write_magic_number(std::ostream& os){
    write_val_to_stream(os, BEGIN_MAGIC_NUMBER);
}

void read_magic_number(std::istream& is){
    int magic = read_val_from_stream<int>(is);
    CHECK(magic == BEGIN_MAGIC_NUMBER);
}

void write_array_to_stream(std::ostream& os, const array& arr) {
    write_magic_number(os);
    write_val_to_stream(os, arr.size());
    write_val_to_stream(os, arr.dim());
    for (int i = 0; i < arr.dim(); ++i) {
        write_val_to_stream(os, arr.dim(i));
    }
    for (int i = 0; i < arr.size(); ++i) {
        write_val_to_stream<float>(os, arr.at(i));
    }
}

array read_array_from_stream(std::istream& is) {
    read_magic_number(is);
    int size = read_val_from_stream<int>(is);
    int dims = read_val_from_stream<int>(is);
    std::vector<int> vdim;
    for (int i = 0; i < dims; ++i) {
        vdim.push_back(read_val_from_stream<int>(is));
    }
    array arr(vdim);
    for (int i = 0; i < size; ++i) {
        arr.at(i) = read_val_from_stream<float>(is);
    }
    return arr;
}



void write_arrays_to_stream(std::ostream& os, const std::vector<array>& arrs) {
    write_magic_number(os);
    write_val_to_stream(os, (int)arrs.size());
    for (int i = 0; i < (int) arrs.size(); ++i){
        write_array_to_stream(os, arrs[i]);
    }
}

std::vector<array> read_arrays_from_stream(std::istream& is) {
    read_magic_number(is);
    int size = read_val_from_stream<int>(is);
    vector<array> arrs;
    for (int i = 0; i < size; ++i){
        array arr = read_array_from_stream(is);
        arrs.push_back(arr);
    }
    return arrs;
}


void write_str_to_stream(std::ostream& os, const std::string& s) {
    write_magic_number(os);
    write_val_to_stream(os, s.size());
    for (int i = 0; i < (int) s.size(); ++i){
        write_val_to_stream(os, s[i]);
    }
}

string read_str_from_stream(std::istream& is) {
    read_magic_number(is);
    int size = read_val_from_stream<int>(is);
    string s(size, 0);
    for (int i = 0; i < size; ++i){
        s[i] = read_val_from_stream<char>(is);
    }
    return s;
}

string read_file(const string& file_name) {
    ifstream is(file_name);
    CHECK(is);
    istreambuf_iterator<char> beg(is), end;
    string txt(beg, end);
    is.close();
    return txt;
}


std::map<std::string, std::shared_ptr<layer> >  build_name_layer_map(
    const std::vector<std::shared_ptr<layer> > &layers){
    map<string, shared_ptr<layer> > layer_map;
    for (auto &layer : layers){
        CHECK(layer_map.count(layer->name()) == 0);
        layer_map[layer->name()] = layer;
    }
    return layer_map;
}

void save_layers(std::ostream& os,
    const std::map<std::string, std::shared_ptr<layer> >& layers){
    for (auto &pair : layers) {
        int head_pos = (int)os.tellp();
        write_val_to_stream<int>(os, 0);
        auto layer = pair.second;
        write_str_to_stream(os,layer->name());
        layer->save(os);
        int tail_pos = (int)os.tellp();
        os.seekp(head_pos);
        write_val_to_stream<int>(os, tail_pos);
        os.seekp(tail_pos);
        cout << "save layer \"" << layer->name() << "\" at " << head_pos << endl;
    }
}

void load_layers(std::istream& is,
    const std::map<std::string, std::shared_ptr<layer> >& layers){
    while (is) {
        int tail_pos = read_val_from_stream<int>(is);
        if (!is) break;
        string name = read_str_from_stream(is);
        if (layers.count(name) != 0){
            auto &layer = layers.find(name)->second;
            cout << "load layer \"" << name << "\"" << endl;
            layer->load(is);
        }
        else{
            cout << "ignore layer \"" << name << "\"" << endl;
        }
        is.seekg(tail_pos);
    }
}

bool yes_no(const string& promote){
    bool bans = true;
    while (true){
        printf("%s(y/n):", promote.c_str());
        string ans;
        cin >> ans;
        if (ans == "y" || ans == "n"){
            bans = (ans == "y");
            break;
        }
    }
    return bans;
}


std::string promote_file_name(const std::string& promote){
    cout << promote << ":";
    string file_name;
    while (true){
        getline(cin, file_name);
        if (!file_name.empty()){
            break;
        }
    }
    if (file_name.size() >= 2 && file_name[0] == '\"') {
        file_name = file_name.substr(1, file_name.size() - 2);
    }
    return file_name;
}

void softmax_normalize(const array& src, array& dst) {
    assert(src.size() == dst.size());
    int sz = src.size();
    const float mmax = src.max();
OMP_FOR
    for (int i = 0; i < sz; ++i) {
        dst.at(i) = exp(src.at(i) - mmax);
    }
    dst.mul(1.0f / dst.sum());
}

void softmax_normalize(const array& src, array2d& dst, int row){
    assert(src.size() == dst.cols());
    assert(row >= 0 && row < dst.rows());
    int sz = src.size();
    const float mmax = src.max();
    //exp
OMP_FOR
    for (int i = 0; i < sz; ++i) {
        dst.at2(row, i) = exp(src.at(i) - mmax);
    }
    //sum
    float ssum = 0;
    for (int i = 0; i < sz; ++i){
        ssum += dst.at2(row, i);
    }
    //normalize
OMP_FOR
    for (int i = 0; i < sz; ++i){
        dst.at2(row, i) /= ssum;
    }
}
