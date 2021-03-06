#ifndef UTILITY_H
#define UTILITY_H

#include "memory.h"
#include "common.h"
#include "layer.h"

array3d imread(const std::string& path);

array3d resize(const array3d& src, int width, int height);

template<class T>
void write_val_to_stream(std::ostream& os, const T t) {
    os.write((char*) &t, sizeof(T));
}

template<class T>
void read_val_from_stream(std::istream& is, T& t) {
    is.read((char*) &t, sizeof(T));
}

template<class T>
T read_val_from_stream(std::istream& is) {
    T t;
    read_val_from_stream(is, t);
    return t;
}

void write_str_to_stream(std::ostream& os, const std::string& s);

std::string read_str_from_stream(std::istream& is);

template<class T1, class T2>
std::vector<T1> convert_arrays(const std::vector<T2>& arrs){
    std::vector<T1> v;
    for (int i = 0; i < (int) arrs.size(); ++i)
        v.push_back(T1(arrs[i]));
    return v;
}


void write_array_to_stream(std::ostream& os, const arraykd& arr);

arraykd read_array_from_stream(std::istream& is);

void write_arrays_to_stream(std::ostream& os, const std::vector<arraykd>& arrs);

std::vector<arraykd> read_arrays_from_stream(std::istream& is);

std::unordered_map<std::string, std::shared_ptr<layer> >  build_name_layer_map(
    const std::vector<std::shared_ptr<layer> > &layers);

void save_layers(std::ostream& os,
    const std::unordered_map<std::string, std::shared_ptr<layer> >& layers);

void load_layers(std::istream& is,
    const std::unordered_map<std::string, std::shared_ptr<layer> >& layers);

std::string read_file(const std::string& file_name);

bool yes_no(const std::string& promote);

std::string promote_file_name(const std::string& promote);

void softmax_normalize(const arraykd& src, arraykd& dst);

void softmax_normalize(const arraykd& src, array2d& dst, int row);

std::vector<std::pair<std::string, std::vector<int> > > read_label_file(const std::string& filename);

#endif