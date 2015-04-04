#include "solver.h"
using namespace std;

void setup_block(const vector<shared_ptr<layer> > &layer_seq){
    for (auto layer : layer_seq) {
        layer->setup_block();
    }
}

void setup_params(const vector<shared_ptr<layer> > &layer_seq){
    for (auto layer : layer_seq) {
        layer->setup_params();
    }
}

void end_batch(const vector<shared_ptr<layer> > &layer_seq, int size) {
    for (auto layer : layer_seq) {
        layer->end_batch(size);
    }
}

void end_batch_and_report_loss(const vector<shared_ptr<layer> > &layer_seq,
    shared_ptr<loss_layer> loss_layer,
    function<void(int epoch)> end_batch_fn,
    int batch, int epoch) {

    //loss
    float loss = loss_layer->loss();
    cout << "epoch " << epoch
        << ", loss " << loss
        << "                                        " << endl;

    //grad
    end_batch(layer_seq, batch);


    //end batch callback
    end_batch_fn(epoch);
}

//layers had been setup
void train(const vector<shared_ptr<layer> > &layer_seq, 
    shared_ptr<loss_layer> loss_layer,
    int batch, function<void(int epoch)> end_batch_fn) {

    for (int iter = 0;; ++iter) {
        int epoch = iter / batch;
        cout << "epoch " << epoch << ", iter: " << iter << "\r";

        //begin
        bool finish = false;
        for (int i = 0; i < (int) layer_seq.size(); ++i){
            auto &layer = layer_seq[i];
            if (!layer->begin_seq()) {
                finish = true;
                break;
            }
        }

        if (finish) {
            if (iter % batch != 0) {
                end_batch_and_report_loss(layer_seq, loss_layer,
                    end_batch_fn, batch, epoch);
            }
            cout << "train finished" << endl;
            break;
        }

        //forward
        for (auto &layer : layer_seq) {
            layer->forward();
        }

        //backward
        for (auto it = layer_seq.rbegin(); it != layer_seq.rend(); ++it){
            (*it)->backward();
        }

        //sgd(every batch times)
        if ((iter + 1) % batch == 0) {
            end_batch_and_report_loss(layer_seq, loss_layer,
                end_batch_fn, batch, epoch);
        }
    }
}
