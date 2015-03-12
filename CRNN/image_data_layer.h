#ifndef IMAGE_DATA_LAYER_H
#define IMAGE_DATA_LAYER_H


#include "array_layer.h"

class image_data_layer : public array_layer {
public:
    image_data_layer(const std::string& dirname,
        const std::string& label_file,
        std::shared_ptr<block> data,
        std::shared_ptr<block> label,
        int batch, int iter,int loop);
};

#endif