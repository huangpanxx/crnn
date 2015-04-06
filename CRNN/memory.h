#ifndef MEMORY_H
#define MEMORY_H

#include "common.h"

inline bool cmp_vec(const std::vector<int> &a, const std::vector<int> &b){
    if (a.size() != b.size()){
        return false;
    }
    for (int i = 0; i < (int) a.size(); ++i){
        if (a[i] != b[i])
            return false;
    }
    return true;
}

struct array_meta {
    float* data;
    int counter;
    int size;
    int id;
    int dim;
    int dimk[20];
};

class arraykd {
public:
    arraykd() : arraykd(0) { };
    arraykd(int size);
    arraykd(const std::vector<int> dims);
    arraykd(const arraykd& b);

    virtual ~arraykd();

    inline int dim() const {
        return this->m_dim;
    }

    inline int dim(int k) const {
        assert(k >= 0 && k < dim());
        return this->m_dimk[k];
    }

    int id() const { return this->m_pmeta->id; }

    std::vector<int> dims() const {
        std::vector<int> ds;
        for (int i = 0; i < this->dim(); ++i){
            ds.push_back(dim(i));
        }
        return ds;
    }
    inline int size() const { return this->m_size; }
    inline float* data() const { return this->m_data; }
    bool operator ==(const arraykd& b) const { return b.data() == data(); }

    inline float& at(int pos) const {
        assert(pos >= 0);
        assert(pos < size());
        return m_data[pos];
    }

    float sum(){
        float sum = 0;
        for (int i = 0; i < this->size(); ++i){
            sum += this->at(i);
        }
        return sum;
    }

    void clear(float val = 0) {
        int sz = this->size();
        OMP_FOR
        for (int i = 0; i < sz; ++i) {
            this->at(i) = val;
        }
    }

    void mul_add(const arraykd& src, float v) {
        assert(src.size() == this->size());
        int sz = this->size();
        OMP_FOR
        for (int i = 0; i < sz; ++i){
            at(i) += src.at(i) * v;
        }
    }

    void add(const arraykd& src) {
        assert(src.size() == this->size());
        OMP_FOR
        for (int i = 0; i < this->size(); ++i){
            at(i) += src.at(i);
        }
    }

    void mul(float v){
        OMP_FOR
        for (int i = 0; i < this->size(); ++i){
            at(i) *= v;
        }
    }

    void rand(float min_val, float max_val) {
        assert(min_val < max_val);
        for (int i = 0; i < this->size(); ++i) {
            float x1 = ::rand() / (float) RAND_MAX;
            float x2 = ::rand() / (float) RAND_MAX;
            float k = (10000 * x1 + x2) / 10001.0f;
            at(i) = k * (max_val - min_val) + min_val;
        }
    }

    arraykd clone(bool is_copy_data = false) {
        //allocate
        arraykd new_arr(this->dims());

        //copy
        if (is_copy_data) {
            new_arr.copy(*this);
        }

        return new_arr;
    }

    arraykd& operator=(const arraykd& b) {
        if (this != &b) {
            destroy();
            this->m_pmeta = b.m_pmeta;
            this->m_pmeta->counter += 1;
            copy_meta();
        }
        return *this;
    };

    void copy(const arraykd& src) {
        copy_data(src, *this);
    }

    float max_val() const{
        int sz = this->size();
        assert(sz > 0);
        float mmax = this->at(0);
        for (int i = 1; i < sz; ++i) {
            mmax = fmax(mmax, at(i));
        }
        return mmax;
    }

    float min_val(){
        int sz = this->size();
        assert(sz > 0);
        float mmin = this->at(0);
        for (int i = 1; i < sz; ++i) {
            mmin = fmin(mmin, at(i));
        }
        return mmin;
    }

    int arg_max(){
        CHECK(size() > 0);
        int k = 0, sz = size();
        for (int i = 1; i < sz; ++i){
            if (at(k) < at(i)) k = i;
        }
        return k;
    }

protected:
    void copy_meta() {
        m_data = m_pmeta->data;
        m_size = m_pmeta->size;
        m_dimk = (int*) (&(m_pmeta->dimk));
        m_dim = m_pmeta->dim;
    }
    array_meta* m_pmeta;
    float *m_data; //backup for speed up
    int *m_dimk;
    int m_dim;
    int m_size;

private:
    void init(int size);
    void destroy();

    static void copy_data(const arraykd& src, arraykd& dst){
        assert(src.size() == dst.size());
        OMP_FOR
        for (int i = 0; i < src.size(); ++i){
            dst.at(i) = src.at(i);
        }
    }

    static int new_id();
};

inline std::ostream& operator <<(std::ostream& os, const arraykd& arr) {
    for (int i = 0; i < arr.size(); ++i){
        if (i != 0) os << ", ";
        os << arr.at(i);
    }
    return os;
}

class array2d : public arraykd {
public:
    array2d(int rows, int cols) : arraykd(rows*cols) {
        this->m_pmeta->dim = 2;
        this->m_pmeta->dimk[0] = rows;
        this->m_pmeta->dimk[1] = cols;
        this->copy_meta();
    }
    array2d(const arraykd& arr) : arraykd(arr) {
        assert(this->dim() == 2);
    }

    inline int rows() const { return dim(0); }
    inline int cols() const { return dim(1); }

    inline float& at2(int row, int col) const {
        int _rows = rows(), _cols = cols();
        assert(row >= 0 && row < _rows);
        assert(col >= 0 && col < _cols);
        return this->at(row * _cols + col);
    }
    int arg_max_row(int row) const {
        assert(size() > 0);
        const int col = cols();
        int k = 0;
        for (int i = 1; i < col; ++i){
            if (at2(row, k) < at2(row, i)) {
                k = i;
            }
        }
        return k;
    }
};

class array3d : public arraykd {
public:
    array3d(int rows, int cols, int channels) : arraykd(rows * cols * channels) {
        this->m_pmeta->dim = 3;
        this->m_pmeta->dimk[0] = rows;
        this->m_pmeta->dimk[1] = cols;
        this->m_pmeta->dimk[2] = channels;
        this->copy_meta();
    }

    array3d(const arraykd& arr) : arraykd(arr) {
        assert(arr.dim() == 3);
    }

    inline float& at3(int row, int col, int channel) const {
        int _rows = rows(), _cols = cols(), _channels = channels();
        assert(row >= 0 && row < _rows);
        assert(col >= 0 && col < _cols);
        assert(channel >= 0 && channel < _channels);
        return this->at(_channels*_cols*row + col*_channels + channel);
    }

    inline int rows() const { return dim(0); }
    inline int cols() const { return dim(1); }
    inline int channels() const { return dim(2); }
}; 

class array4d : public arraykd {
public:
    array4d() : arraykd(0){
        this->m_pmeta->dim = 4;
        this->m_pmeta->dimk[0] = 0;
        this->m_pmeta->dimk[1] = 0;
        this->m_pmeta->dimk[2] = 0;
        this->m_pmeta->dimk[3] = 0;
        this->copy_meta();
    }

    array4d(int nums, int rows, int cols, int channels)
        : arraykd(rows * cols * channels * nums) {
        this->m_pmeta->dim = 4;
        this->m_pmeta->dimk[0] = nums;
        this->m_pmeta->dimk[1] = rows;
        this->m_pmeta->dimk[2] = cols;
        this->m_pmeta->dimk[3] = channels;
        this->copy_meta();
    }

    array4d(const arraykd& arr) : arraykd(arr) {
        assert(arr.dim() == 4);
    }

    inline float& at4(int num, int row, int col, int channel) const {
        int _rows = rows(), _cols = cols(), _channels = channels(), _nums = this->nums();
        assert(row >= 0 && row < _rows);
        assert(col >= 0 && col < _cols);
        assert(channel >= 0 && channel < _channels);
        assert(num >= 0 && num < _nums);
        //this is slow
        return this->at(num*_rows*_cols*_channels + row*_cols*_channels + col*_channels + channel);
    }

    inline int rows() const { return dim(0); }
    inline int cols() const { return dim(1); }
    inline int channels() const { return dim(2); }
    inline int nums() const { return dim(3); }
};


//b & c are vertical vectors
inline void mul_addv(const array2d& a, const arraykd&b, arraykd& c){
    CHECK(a.rows() == c.size());
    CHECK(a.cols() == b.size());
    int row = a.rows(), col = a.cols();
    OMP_FOR
    for (int i = 0; i < row; ++i){
        float s = 0;
        for (int j = 0; j < col; ++j){
            s += a.at2(i, j) * b.at(j);
        }
        c.at(i) += s;
    }
}

//a & c are horizontal vectors
inline void mul_addh(const arraykd& a, const array2d&b, arraykd& c){
    assert(a.size() == b.rows());
    assert(c.size() == b.cols());
    int row = b.rows(), col = b.cols();
    OMP_FOR
    for (int i = 0; i < col; ++i){
        float s = 0;
        for (int j = 0; j < row; ++j){
            s += a.at(j) * b.at2(j, i);
        }
        c.at(i) += s;
    }
}

inline void mul_wise(const arraykd& a, const arraykd& b, arraykd& c){
    assert(a.size() == b.size());
    assert(a.size() == c.size());
    const int size = a.size();
    OMP_FOR
    for (int i = 0; i < size; ++i) {
        c.at(i) = a.at(i) * b.at(i);
    }
}

inline void mul_wise_add(const arraykd& a, const arraykd& b, arraykd& c){
    assert(a.size() == b.size());
    assert(a.size() == c.size());
    const int size = a.size();
    OMP_FOR
    for (int i = 0; i < size; ++i) {
        c.at(i) += a.at(i) * b.at(i);
    }
}

inline void mul(const arraykd& src, float factor, arraykd& dst){
    assert(src.size() == dst.size());
    const int size = src.size();
    OMP_FOR
    for (int i = 0; i < size; ++i) {
        dst.at(i) = src.at(i) * factor;
    }
}

inline void mul(const arraykd& src, float factor, float bias, arraykd& dst) {
    assert(src.size() == dst.size());
    const int size = src.size();
    OMP_FOR
    for (int i = 0; i < size; ++i) {
        dst.at(i) = src.at(i) * factor + bias;
    }
}

inline void mul_add(const arraykd& src, float factor, arraykd& dst){
    assert(src.size() == dst.size());
    const int size = src.size();
    OMP_FOR
    for (int i = 0; i < size; ++i) {
        dst.at(i) += src.at(i) * factor;
    }
}


inline void add_to(const arraykd& src, arraykd& dst){
    assert(src.size() == dst.size());
    const int sz = src.size();
    OMP_FOR
    for (int i = 0; i < sz; ++i){
        dst.at(i) += src.at(i);
    }
}


class block{
public:
    static std::shared_ptr<block> new_block() {
        return std::shared_ptr<block>(new block());
    }
public:
    block(){};

    void resize(int size) {
        m_signal = arraykd(size);
        m_error = arraykd(size);
    }

    void resize(int rows, int cols) {
        m_signal = array2d(rows, cols);
        m_error = array2d(rows, cols);
    }

    void resize(int rows, int cols, int channels){
        m_signal = array3d(rows, cols, channels);
        m_error = array3d(rows, cols, channels);
    }

    void resize(const std::vector<int> &dims){
        m_signal = arraykd(dims);
        m_error = arraykd(dims);
    }

    void clear(float v = 0) {
        this->signal().clear(v);
        this->error().clear(v);
    }

    // never set the m_signal & m_error variable directly,
    // unless make sure src&dst have the same dims
    arraykd& signal(){ return m_signal; }
    arraykd& error(){ return m_error; }

    inline void set_signal(arraykd& src) {
        CHECK(cmp_vec(src.dims(), m_signal.dims()));
        this->m_signal = src;
    }

    arraykd& new_signal() {
        m_signal = m_signal.clone(false);
        return m_signal;
    }

    std::vector<int> dims() {
        return this->m_signal.dims();
    }

    int size(){
        return this->m_signal.size();
    }

    bool empty(){
        return this->m_signal.size() == 0;
    }

private:
    arraykd m_signal;
    arraykd m_error;
};

typedef std::shared_ptr<block> block_ptr;


class block_factory {
public:
    block_ptr get_block(const std::string& id);

    std::vector<block_ptr> get_blocks(const std::vector<std::string>& ids);

    bool contains(const std::string& id) {
        return m_cache.count(id) != 0;
    }

    static std::shared_ptr<block_factory> new_factory() {
        return std::shared_ptr<block_factory>(new block_factory());
    }

private:
    std::unordered_map<std::string, block_ptr> m_cache;
};



inline bool cmp_array_dim(const arraykd& a, const arraykd& b){
    if (a.dim() != b.dim()) return false;
    return cmp_vec(a.dims(), b.dims());
}


#endif