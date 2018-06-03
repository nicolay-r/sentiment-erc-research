#!/usr/bin/python
# -*- coding: utf-8 -*-

from core.classification import evaluate, fit_and_predict, filter_features, \
    create_train_data, create_test_data, create_model_feature_importances, \
    plot_feature_importances

import io_utils
from classifiers.rfe import get_rfe


#
# MAIN
#
synonyms_filepath = io_utils.get_synonyms_filepath()

print "Preparing Train Collection"
X_train, y_train = create_train_data(io_utils.get_train_vectors_list())
print "Preparing Test Collection"
X_test, test_collections = create_test_data(io_utils.get_test_vectors_list())

RFE = get_rfe(X_train, y_train)

# features selection RFE
for model_name in RFE:
    model = RFE[model_name]['model']
    estimator = RFE[model_name]['estimator']
    model.fit(X_train, y_train)
    idf = create_model_feature_importances(
            model.estimator,
            model_name,
            model.get_support(),
            io_utils.read_feature_names())
    print "Feature selection for RFE method: {}".format(model_name)
    X_train_new, X_test_new = filter_features(model_name, model, X_train, X_test)
    test_opinions = fit_and_predict(model_name,
                                    estimator,
                                    X_train_new,
                                    y_train,
                                    X_test_new,
                                    test_collections,
                                    synonyms_filepath)
    io_utils.save_test_opinions(test_opinions, model_name)

    edf = evaluate(estimator,
                   model_name,
                   io_utils.create_files_to_compare_list(model_name),
                   io_utils.get_method_root(model_name),
                   synonyms_filepath)

    e_filepath = "{}/test_{}.csv".format(io_utils.eval_rfe_root(), model_name)
    i_filepath = "{}/test_{}.features.csv".format(io_utils.eval_features_filepath(), model_name)
    fig_filepath = '{}/test_{}.png'.format(io_utils.eval_root(), model_name)

    edf.to_csv(e_filepath)
    idf.to_csv(i_filepath)
    fig = plot_feature_importances(i_filepath)
    fig.savefig(fig_filepath)
