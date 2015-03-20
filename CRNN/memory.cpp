#include "memory.h"
#include <mutex>
using namespace std;

struct mem_array {
    mem_array(int size, float* pointer) {
        this->size = size;
        this->pointer = pointer;
        this->occupied = false;
    }
    int size;
    float* pointer;
    bool occupied;
};

mutex& get_mem_mutex(){
    static mutex mem_mutex;
    return mem_mutex;
}

vector<mem_array>& get_mem_arrays(){
    static vector<mem_array> mem_arrays;
    return mem_arrays;
}

int &get_active_arrays(){
    static int active_arrays = 0;
    return active_arrays;
}


float* alloc_array(int size) {
    auto& g_mem_mutex = get_mem_mutex();
    auto& g_active_arrays = get_active_arrays();
    auto& g_arrays = get_mem_arrays();
    g_mem_mutex.lock();
    float* pt = 0;
    g_active_arrays += 1;
    for (int i = 0; i < (int) g_arrays.size(); ++i) {
        auto &array = g_arrays[i];
        //bigger than size and less than 2 times size
        if (!array.occupied && array.size >= size && array.size <= size * 2) {
            pt = array.pointer;
            array.occupied = true;
            //cout << "reuse array, size = " << size << ", actual = "<< array.size << endl;
            break;
        }
    }
    if (0 == pt) {
        pt = new float[size];
        mem_array array(size, pt);
        array.occupied = true;
        //cout << "new array, size = " << array.size << endl;
        for (int i = 0; i < (int) g_arrays.size() + 1; ++i) {
            if (i == g_arrays.size() || g_arrays[i].size >= size) {
                g_arrays.insert(g_arrays.begin() + i, array);
                break;
            }
        }
    }
    g_mem_mutex.unlock();
    return pt;
}

void free_array(float* pt) {
    auto& g_mem_mutex = get_mem_mutex();
    auto& g_active_arrays = get_active_arrays();
    auto& g_arrays = get_mem_arrays();
    g_mem_mutex.lock();
    g_active_arrays -= 1;
    for (int i = 0; i < (int) g_arrays.size(); ++i) {
        auto &array = g_arrays[i];
        if (array.pointer == pt) {
            //cout << "free array, size = " << array.size 
            //<<", active = " << g_active_arrays << endl;
            array.occupied = false;
            break;
        }
    }
    g_mem_mutex.unlock();
}

int array::new_id(){
    auto& g_mem_mutex = get_mem_mutex();
    static int sid = -1;
    g_mem_mutex.lock();
    ++sid;
    g_mem_mutex.unlock();
    return sid;
}


array::array(int size) {
    init(size);
    this->m_pmeta->dim = 1;
    this->m_pmeta->dimk[0] = size;
    copy_meta();
}

array::array(const std::vector<int> dims) {
    assert(dims.size() != 0);
    assert(dims.size() < 20);
    int size = 1;
    for (int i = 0; i < (int) dims.size(); ++i){
        size *= dims[i];
    }
    init(size);
    this->m_pmeta->dim = (int) dims.size();
    for (int i = 0; i < this->m_pmeta->dim; ++i){
        this->m_pmeta->dimk[i] = (int) dims[i];
    }
    copy_meta();
}



void array::init(int size) {
    const int offset = sizeof(array_meta) / sizeof(float) +1;
    float* pmem = alloc_array(size + offset);
    this->m_pmeta = (array_meta*) pmem;
    this->m_pmeta->data = pmem + offset;
    this->m_pmeta->id = array::new_id();
    this->m_pmeta->size = size;
    this->m_pmeta->counter = 1;
}

array::array(const array& arr) {
    auto& g_mem_mutex = get_mem_mutex();
    this->m_pmeta = arr.m_pmeta;
    g_mem_mutex.lock();
    this->m_pmeta->counter += 1;
    g_mem_mutex.unlock();
    this->copy_meta();
}

array::~array() {
    destroy();
}

void array::destroy(){
    auto& g_mem_mutex = get_mem_mutex();
    g_mem_mutex.lock();
    this->m_pmeta->counter -= 1;
    g_mem_mutex.unlock();
    if (this->m_pmeta->counter == 0) {
        free_array((float*)this->m_pmeta);
    }
}


block_ptr block_factory::get_block(int id) {
    if (m_cache.count(id) == 0) {
        m_cache[id] = block::new_block();
    }
    return m_cache[id];
}

vector<block_ptr> block_factory::get_blocks(const vector<int>& ids){
    vector<block_ptr> blocks;
    for (int id : ids){
        blocks.push_back(this->get_block(id));
    }
    return blocks;
}
