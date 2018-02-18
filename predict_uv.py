#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import feature_selection
from sklearn.feature_selection import chi2, f_classif

from core.classification import evaluate, fit_and_predict, \
    create_train_data, create_test_data, create_model_feature_importances

import io_utils

from classifiers.estimators_all import CLASSIFIERS


#
# MAIN
#
synonyms_filepath = io_utils.get_synonyms_filepath()

UNIVARIATE = {
    "uv_kbest_def": feature_selection.SelectKBest(f_classif, k=10),
    "uv_kbest_chi2_def": feature_selection.SelectKBest(chi2, k=10),
    "uv_percentile_def": feature_selection.SelectPercentile(f_classif, percentile=10),
    "uv_fpr_def": feature_selection.SelectFpr(f_classif),
    "uv_fwe_def": feature_selection.SelectFwe(f_classif)
}

print "Preparing Train Collection"
X_train, y_train = create_train_data(io_utils.get_train_vectors_list())
print "Preparing Test Collection"
X_test, test_collections = create_test_data(io_utils.get_train_vectors_list())

# Univariate
for univariate_model_name in UNIVARIATE:
    model = UNIVARIATE[univariate_model_name]
    model.fit(X_train, y_train)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    for method_name in CLASSIFIERS:
        name = "{}_{}".format(method_name, univariate_model_name)
        print "Evaluate for univariate estimator: {}".format(name)
        test_opinions = fit_and_predict(
                name, CLASSIFIERS[method_name], X_train_new, y_train,
                X_test_new, test_collections, synonyms_filepath)
        io_utils.save_test_opinions(test_opinions, name)

        idf = create_model_feature_importances(
                None,
                name,
                model.get_support(),
                io_utils.read_feature_names())
        edf = evaluate(CLASSIFIERS[method_name],
                       name,
                       io_utils.create_files_to_compare_list(method_name),
                       io_utils.get_method_root(method_name),
                       synonyms_filepath)
        edf.to_csv("{}/test_{}.csv".format(io_utils.eval_univariate_root(), method_name))
        idf.to_csv("{}/test_{}.features.csv".format(io_utils.eval_features_filepath(), method_name))
