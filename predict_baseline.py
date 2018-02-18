#!/usr/bin/python
# -*- coding: utf-8 -*-

from core.classification import evaluate, fit_and_predict, \
    create_train_data, create_test_data

import io_utils
from classifiers import estimators

#
# MAIN
#
synonyms_filepath = io_utils.get_synonyms_filepath()

print "Preparing Train Collection"
X_train, y_train = create_train_data(io_utils.get_train_vectors_list())
print "Preparing Test Collection"
X_test, test_collections = create_test_data(io_utils.get_test_vectors_list())

BASELINES = {
    'baseline_pos': estimators.baseline_pos,
    'baseline_neg': estimators.baseline_neg,
    'baseline_rand': estimators.baseline_rand,
    'baseline_strat': estimators.baseline_strat,
}

# baseline estimators
for method_name in BASELINES:
    print "Evaluate for baseline estimator: {}".format(method_name)
    test_opinions = fit_and_predict(
            method_name, BASELINES[method_name], X_train, y_train, X_test,
            test_collections, synonyms_filepath)
    io_utils.save_test_opinions(test_opinions, method_name)

    edf = evaluate(BASELINES[method_name],
                   method_name,
                   io_utils.create_files_to_compare_list(method_name),
                   io_utils.get_method_root(method_name),
                   synonyms_filepath)

    edf.to_csv("{}/test_{}.csv".format(
        io_utils.eval_baseline_root(),
        method_name))
