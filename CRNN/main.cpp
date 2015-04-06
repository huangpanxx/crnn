#include "test.h"
#include "utility.h"
#include "array_operator.h"
using namespace std;

void test_code(){
    auto op = get_default_array_operator();
    int m = 12500, n = 12500;
    array2d a(m, n);
    arraykd b(n);
    arraykd c(m);

    cout << "cpu" << endl;
    TIME("cpu", [=, &a, &b, &c](){
        mul_addv(a, b, c);
    });

    cout << "gpu" << endl;
    TIME("gpu", [=, &a, &b, &c]() {
        op->mul_addv(a, b, c);
    });
}

void test(){
    test_code();
    system("pause");
    exit(0);
}

int main(int argc, char **argv) {
    test();

    string model_file = "";
    if (argc == 2) {
        model_file = argv[1];
    }
    if (argc == 1) {
        model_file = promote_file_name("model config file(*.json)");
    }
    train_and_test_network(model_file);
    system("pause");
    return 0;
}
