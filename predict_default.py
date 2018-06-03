#!/usr/bin/python
# -*- coding: utf-8 -*-

from core.classification import evaluate, fit_and_predict, create_train_data, \
    create_test_data

import io_utils
from classifiers.estimators_all import CLASSIFIERS


#
# MAIN
#
synonyms_filepath = io_utils.get_synonyms_filepath()

print "Preparing Train Collection"
X_train, y_train = create_train_data(io_utils.get_train_vectors_list())
print "Preparing Test Collection"
X_test, test_collections = create_test_data(io_utils.get_test_vectors_list())


# classfiers with predefined settings
for method_name in CLASSIFIERS:
    print "Evaluate for default estimator: {}".format(method_name)
    test_opinions = fit_and_predict(
            method_name, CLASSIFIERS[method_name], X_train, y_train, X_test,
            test_collections, synonyms_filepath)
    io_utils.save_test_opinions(test_opinions, method_name)

    edf = evaluate(CLASSIFIERS[method_name],
                   method_name,
                   io_utils.create_files_to_compare_list(method_name),
                   io_utils.get_method_root(method_name),
                   synonyms_filepath)
    edf.to_csv("{}/test_{}.csv".format(io_utils.eval_default_root(), method_name))
