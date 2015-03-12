#ifndef SOLVER_H
#define SOLVER_H

#include "common.h"
#include "layer.h"

void setup_block(const std::vector<std::shared_ptr<layer> > &layer_seq);
void setup_params(const std::vector<std::shared_ptr<layer> > &layer_seq);
void train(const std::vector<std::shared_ptr<layer> > &layer_seq,
    std::shared_ptr<loss_layer> loss_layer,
    int batch, std::function<void(int)> end_batch_fn = [](int epoch){});

#endif