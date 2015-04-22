#ifndef IMAGE_SPLIT_LAYER_H
#define IMAGE_SPLIT_LAYER_H

#include "layer.h"

class image_split_helper {
public:
    image_split_helper(
        const array3d& image,
        int width, int height, 
        int stride,
        int leftshift,
        int rightshift,
        int topshift);

    image_split_helper() : m_image(0, 0, 0){ };

    array3d image_slice(int k);

    int image_slice_num();

private:
    array3d m_image;
    int m_stride;
    int m_height;
    int m_width;
    int m_leftshift, m_rightshift, m_topshift;
};


class image_slice_layer : public feed_data_layer {
public:
    image_slice_layer(block_ptr data_block,
        int width, int height,
        int stride_min, int stride_max,
        int left_shift_min, int left_shift_max,
        int right_shift_min, int right_shift_max,
        int top_shift_min, int top_shift_max);
    virtual void setup_block();
    virtual bool begin_seq();
    virtual bool forward();
    void set_data(const arraykd& image);
    
private:
    image_split_helper m_helper;
    block_ptr m_data_block;
    int m_t;
    int m_width, m_height;
    int m_stride_min, m_stride_max;
    int m_left_shift_min, m_left_shift_max;
    int m_right_shift_min, m_right_shift_max;
    int m_top_shift_min, m_top_shift_max;
};

class label_slice_layer : public feed_label_layer {
public:
    label_slice_layer(block_ptr label_block, int label_size);
    virtual void setup_block();
    virtual bool begin_seq();
    virtual bool forward();
    virtual void backward();
    void set_label(const std::vector<int> &labels);

private:
    block_ptr m_label_block;
    std::vector<int> m_labels;
    int m_t;
    int m_label_size;
};

class image_split_layer : public data_layer {
public:
    image_split_layer(
        const std::string& label_file,
        const std::string& data_dir,
        int width, int height,
        int stride_min,int stride_max,
        int leftshift_min, int leftshift_max,
        int rightshift_min, int rightshift_max,
        int topshift_min,int topshift_max,
        int batch, int label_size,
        block_ptr data_block,
        block_ptr label_block);

    virtual void setup_block();
    virtual void setup_params();
    virtual bool begin_seq();
    virtual bool forward();
    virtual void save(std::ostream& os);
    virtual void load(std::istream& is);

    virtual int batch();
    virtual void move_to_next_batch();

    layer_ptr get_image_slice_layer(){
        return m_image_slice_layer;
    }

    layer_ptr get_label_slice_layer(){
        return m_label_slice_layer;
    }

private:
    long m_index;
    int m_batch, m_label_size;
    int m_width, m_height, m_stride_min, m_stride_max;
    std::vector<std::pair<std::string, std::vector<int> > > m_samples;
    layer_ptr m_image_slice_layer;
    layer_ptr m_label_slice_layer;
    block_ptr m_data_block, m_label_block;
    std::string m_data_dir;
};

#endif