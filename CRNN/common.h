#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <cassert>
#include <vector>
#include <queue>
#include <memory>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <functional>
#include <map>
#include <ctime>

#ifndef _DEBUG

#undef assert
#define assert(x)
#define OMP_FOR __pragma(omp parallel for)
//#define OMP_FOR

#else

#define OMP_FOR

#endif




inline void print_check(const char* msg, const char* filename, int line) {
    std::cerr
        << "\"" << filename << "\", line " << line 
        << ": CHECK \"" << msg << "\" failed!" << std::endl;
    system("pause");
    exit(0);
}

#define CHECK(_Expression) \
    (void)((!!(_Expression)) || (print_check(#_Expression,__FILE__, __LINE__), 0) )

#endif