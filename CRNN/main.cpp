#include <iostream>
#include <omp.h>
#include "memory.h"
#include "utility.h"
#include "picojson.h"
#include "layer.h"
#include "array_layer.h"
#include "solver.h"
#include "inner_product_layer.h"
#include "softmax_loss_layer.h"
#include "sigmoid_layer.h"
#include "image_data_layer.h"
#include "conv_layer.h"
#include "relu_layer.h"
#include "softmax_layer.h"
#include "network.h"
using namespace std;


void test_xor() {
    array s1(2), s2(2), s3(2), s4(2);
    array2d l1(1, 2), l2(1, 2), l3(1, 2), l4(1, 2);
    s1.at(0) = 0, s1.at(1) = 1; l1.at(0) = 0; l1.at(1) = 1;
    s2.at(0) = 1, s2.at(1) = 0; l2.at(0) = 0; l2.at(1) = 1;
    s3.at(0) = 0, s3.at(1) = 0; l3.at(0) = 1; l3.at(1) = 0;
    s4.at(0) = 1, s4.at(1) = 1; l4.at(0) = 1; l4.at(1) = 0;
    vector<array_sample> samples;
    samples.push_back(array_sample(s1, l1));
    samples.push_back(array_sample(s2, l2));
    samples.push_back(array_sample(s3, l3));
    samples.push_back(array_sample(s4, l4));

    shared_ptr<block> data_block(new block());
    shared_ptr<block> label_block(new block());
    shared_ptr<block> ip_hid1_block(new block());
    shared_ptr<block> sig_hid1_block(new block());
    shared_ptr<block> ip_hid2_block(new block());

    shared_ptr<array_layer> array_layer(new array_layer(samples, data_block,
        label_block, 4, 50000, 1));
    shared_ptr<layer> ip1_layer(new inner_product_layer(2, data_block, ip_hid1_block));
    shared_ptr<layer> sig1_layer(new relu_layer(ip_hid1_block, sig_hid1_block));
    shared_ptr<layer> ip2_layer(new inner_product_layer(2, sig_hid1_block, ip_hid2_block));
    shared_ptr<softmax_loss_layer> loss_layer(new softmax_loss_layer(ip_hid2_block, label_block));

    loss_layer->set_report(false);

    ip1_layer->set_learn_rate(0.1f);
    ip2_layer->set_learn_rate(0.1f);

    ip1_layer->set_momentum_decay(0.9f);
    ip2_layer->set_momentum_decay(0.9f);

    vector<shared_ptr<layer> > layer_seq;
    layer_seq.push_back(array_layer);
    layer_seq.push_back(ip1_layer);
    layer_seq.push_back(sig1_layer);
    layer_seq.push_back(ip2_layer);
    layer_seq.push_back(loss_layer);

    setup_block(layer_seq);
    setup_params(layer_seq);
    train(layer_seq, loss_layer, array_layer->batch());
}

void test_mlp(){
    auto data_block = block::new_block();
    auto label_block = block::new_block();
    auto fc1_block = block::new_block();
    auto sig1_block = block::new_block();
    auto fc2_block = block::new_block();

    shared_ptr<image_data_layer> data_layer(
        new image_data_layer(
        "data",
        "label.txt",
        data_block, label_block, 15, 10000, 100, -1, -1));

    shared_ptr<inner_product_layer> fc1_layer(
        new inner_product_layer(200, data_block , fc1_block));

    shared_ptr<layer> sig1_layer(
        new sigmoid_layer(fc1_block, sig1_block));

    shared_ptr<inner_product_layer> fc2_layer(
        new inner_product_layer(36, sig1_block, fc2_block));

    shared_ptr<softmax_loss_layer> loss_layer(
        new softmax_loss_layer(fc2_block, label_block));



    vector<shared_ptr<layer> > layer_seq;
    layer_seq.push_back(data_layer);
    layer_seq.push_back(fc1_layer);
    layer_seq.push_back(sig1_layer);
    layer_seq.push_back(fc2_layer);
    layer_seq.push_back(loss_layer);

    for (int i = 0; i < (int) layer_seq.size(); ++i){
        layer_seq[i]->set_learn_rate(0.01f);
    }

    loss_layer->set_report(false);

    setup_block(layer_seq);
    setup_params(layer_seq);
    train(layer_seq, loss_layer, data_layer->batch());
}

void test_conv() {
    auto data_block = block::new_block();
    auto label_block = block::new_block();
    auto conv1_block = block::new_block();
    auto sig1_block = block::new_block();
    auto conv2_block = block::new_block();
    auto sig2_block = block::new_block();
    auto fc1_block = block::new_block();
    auto sig3_block = block::new_block();
    auto fc2_block = block::new_block();
    auto output_block = block::new_block();

    shared_ptr<image_data_layer> data_layer(
        new image_data_layer(
        "data",
        "label.txt",
        data_block, label_block, 200, 30, 300));
    data_layer->set_name("data");

    shared_ptr<conv_layer> conv1_layer(
        new conv_layer(data_block, conv1_block, 5, 20, 3));
    conv1_layer->set_name("conv1");

    shared_ptr<layer> sig1_layer(
        new relu_layer(conv1_block, sig1_block));
    sig1_layer->set_name("relu1");

    shared_ptr<conv_layer> conv2_layer(
        new conv_layer(sig1_block, conv2_block, 5, 30, 4));
    conv2_layer->set_name("conv2");

    shared_ptr<layer> sig2_layer(
        new relu_layer(conv2_block, sig2_block));
    sig2_layer->set_name("relu2");

    shared_ptr<inner_product_layer> fc1_layer(
        new inner_product_layer(800, sig2_block, fc1_block));
    fc1_layer->set_name("fc1.1");

    shared_ptr<layer> sig3_layer(
        new relu_layer(fc1_block, sig3_block));
    sig3_layer->set_name("relu3");


    int output = data_layer->label_dims()[1];
    shared_ptr<inner_product_layer> fc2_layer(
        new inner_product_layer(output, sig3_block, fc2_block));
    fc2_layer->set_name("fc2.1");

    shared_ptr<softmax_loss_layer> loss_layer(
        new softmax_loss_layer(fc2_block, label_block));
    loss_layer->set_name("loss");

    shared_ptr<softmax_layer> output_layer(
        new softmax_layer(fc2_block, output_block));
    output_layer->set_name("output");

    vector<shared_ptr<layer> > train_layer_seq;
    train_layer_seq.push_back(data_layer);
    train_layer_seq.push_back(conv1_layer);
    train_layer_seq.push_back(sig1_layer);
    train_layer_seq.push_back(conv2_layer);
    train_layer_seq.push_back(sig2_layer);
    train_layer_seq.push_back(fc1_layer);
    train_layer_seq.push_back(sig3_layer);
    train_layer_seq.push_back(fc2_layer);
    train_layer_seq.push_back(loss_layer);

    vector<shared_ptr<layer> > pred_layer_seq;
    pred_layer_seq.push_back(conv1_layer);
    pred_layer_seq.push_back(sig1_layer);
    pred_layer_seq.push_back(conv2_layer);
    pred_layer_seq.push_back(sig2_layer);
    pred_layer_seq.push_back(fc1_layer);
    pred_layer_seq.push_back(sig3_layer);
    pred_layer_seq.push_back(fc2_layer);
    pred_layer_seq.push_back(output_layer);

    //load
    string model_file = "model.bin";
    ifstream fin(model_file, ios::in | ios::binary);
    if (fin) {
        load_layers(fin, build_name_layer_map(train_layer_seq));
    }
    fin.close();


    bool is_train = true;
    while (true){
        cout << "y for predict, n for train(y/n):";
        string ans;
        cin >> ans;
        if (ans == "y" || ans == "n"){
            is_train = (ans == "n");
            break;
        }
    }

    if (is_train){
        for (int i = 0; i < (int) train_layer_seq.size(); ++i) {
            train_layer_seq[i]->set_learn_rate(0.015f + (::rand() / RAND_MAX)*0.01f);
            train_layer_seq[i]->set_momentum_decay(8.0f);
        }
        //loss_layer->set_report(true);
        //train
        setup_block(train_layer_seq);
        setup_params(train_layer_seq);
        train(train_layer_seq, loss_layer, data_layer->batch(),
            [&model_file, &train_layer_seq, &loss_layer, &data_layer](int epoch) {
            const int snapshot = 20;
            //save
            if (epoch % snapshot == (snapshot - 1)) {
                cout << "save model ..." << endl;
                ofstream fout(model_file, ios::out | ios::binary);
                save_layers(fout, build_name_layer_map(train_layer_seq));
            }
            if (loss_layer->loss() <= 0.08f) {
                cout << "move to next batch" << endl;
                data_layer->move_to_next_batch();
                for (auto layer : train_layer_seq) {
                    layer->set_learn_rate(0.01f);
                }
            }
        });
    }
    else {
        while (true){
            cout << "image path:";
            string img_path;
            cin >> img_path;
            CHECK(!img_path.empty());

            array3d img = imread(img_path);
            CHECK(img.size() != 0);
            data_block->resize(img.dims());
            data_block->signal() = img;
            setup_block(pred_layer_seq);
            setup_params(pred_layer_seq);
            for (auto layer : pred_layer_seq){
                layer->begin_seq();
            }
            for (auto layer : pred_layer_seq){
                layer->forward(0);
            }
            auto &signal = output_block->signal();
            cout << signal << endl;
            int label = 0;
            for (int i = 0; i < signal.size(); ++i) {
                if (signal.at(i)>signal.at(label)) {
                    label = i;
                }
            }
            cout << "label: " << label << endl;
        }
    }
}

void test_stream(){
    string model_file = "C:/Users/snail/Desktop/test.bin";
    ofstream fout(model_file, ios::out | ios::binary);
    write_str_to_stream(fout, "asd");
    fout.close();

    ifstream fin(model_file, ios::in | ios::binary);
    string s = read_str_from_stream(fin);
}

void test_omp(){
    int s[1000];
    for (int i = 0; i < 1000; ++i){
        int sum = 0;
#pragma omp for
        for (int k = 0; k < 100000; ++k){
            sum += 1;
        }
        s[i] = sum;
    }
    for (int i = 0; i < 1000; ++i){
        if (s[i] != 100000){
            cout << "xx" << endl;
        }
    }
}

void test_network(const string& filename) {
    printf("load file %s.\n", filename.c_str());
    string json = read_file(filename);

    bool is_train = yes_no("train or predict");
    if (is_train){
        network train_net(json, "train");
        train_net.train();
    }
    else{
        network predict_net(json, "predict");

        while (true) {
            //read image
            auto file_name = promote_file_name("image file");
            auto image = imread(file_name);
            auto dims = predict_net.input_dims();

            //resize image
            CHECK(dims.size() == 3 && dims[2] == 3);
            int width = dims[1], height = dims[0];
            if (width != image.cols() || height != image.rows()){
                image = resize(image, width, height);
            }

            //recongnize
            int start_time = clock();
            predict_net.set_input(image);
            array _output = predict_net.forward();

            //get 2d output
            array2d output(0, 0);
            if (_output.dim() == 1) {
                output = array2d(1, _output.size());
                output.copy(_output);
            }
            else if (_output.dim() == 2) {
                output = _output;
            }
            else{ CHECK(false); }

            //translate
            string ans = "";
            for (int i = 0; i < output.rows(); ++i) {
                int k = output.arg_max_row(i);
                ans += predict_net.translate(k);
                printf("%.3f ", output.at2(i, k));
            }
            printf("\n");

            //print info
            float freq = 1.0f * CLOCKS_PER_SEC / (clock() - start_time);
            printf("result: %s, speed %.3f/s\n", ans.c_str(), freq);
        }
    }
}

void test(){
    for (int i = 0; i < 99999; ++i){
        float v = float(i);
        int k = (int)ceil(v);
        if (v != i) {
            printf("%d ", i);
            system("pause");
        }
    }

}

int main(int argc, char **argv) {
    //test_stream();
    //test_conv();
    //test_xor();
    //test_mlp();
    //test_omp();

    string model_file = "";
    if (argc == 2) {
        model_file = argv[1];
    }
    if (argc == 1) {
        model_file = promote_file_name("model config file(*.json)");
    }
    test_network(model_file);
    system("pause");
    return 0;
}
