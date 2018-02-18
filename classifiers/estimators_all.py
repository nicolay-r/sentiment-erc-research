import estimators

CLASSIFIERS = {
    'svm': estimators.e_svm,
    'rf': estimators.e_rf,
    'nb-gaussian': estimators.e_nb_gaussian,
    'nb-bernoulli': estimators.e_nb_bernoulli,
    'knn': estimators.e_knn,
    'nb-bernoulli-grid': estimators.e_nb_bernoulli_grid,
    'knn-grid': estimators.e_knn_grid,
    'rf-grid': estimators.e_rf_grid,
    'svm-grid': estimators.e_svm_grid
}
