import numpy as np
from sklearn import svm, neighbors, ensemble, model_selection, naive_bayes, dummy

# Baseline estimators
baseline_pos = dummy.DummyClassifier(
        strategy='constant', constant=1, random_state=0)
baseline_neg = dummy.DummyClassifier(
        strategy='constant', constant=-1, random_state=0)
baseline_rand = dummy.DummyClassifier(
        strategy='uniform', random_state=0)
baseline_strat = dummy.DummyClassifier(
        strategy='stratified', random_state=0)

# Estimators with predefined settings
e_svm = svm.SVC(C=1,
                kernel='linear',
                cache_size=1000,
                class_weight='balanced',
                random_state=0)
e_rf = ensemble.RandomForestClassifier(n_estimators=10,
                                       class_weight="balanced",
                                       random_state=0)
e_nb_gaussian = naive_bayes.GaussianNB()

e_nb_bernoulli = naive_bayes.BernoulliNB()

e_knn = neighbors.KNeighborsClassifier()

# Grid search
cv_count = 10
f1_macro_scoring = 'f1_macro'
n_jobs = 1

e_nb_bernoulli_grid = model_selection.GridSearchCV(
        naive_bayes.BernoulliNB(),
        {
            'alpha': np.arange(0.0, 1, 0.1),
            'binarize': np.arange(0.0, 1, 0.1),
        },
        scoring=f1_macro_scoring,
        cv=cv_count,
        n_jobs=n_jobs)

e_knn_grid = model_selection.GridSearchCV(
        neighbors.KNeighborsClassifier(),
        {
            'n_neighbors': np.arange(1, 10),
        },
        scoring=f1_macro_scoring,
        cv=cv_count,
        n_jobs=n_jobs)

e_svm_grid = model_selection.GridSearchCV(
        svm.SVC(cache_size=1000, random_state=0),
        {
            'C': np.arange(0.2, 5, 2),
            'class_weight': ['balanced'],
            'kernel': ['linear']
        },
        scoring=f1_macro_scoring,
        cv=cv_count,
        n_jobs=n_jobs)

e_rf_grid = model_selection.GridSearchCV(
        ensemble.RandomForestClassifier(random_state=0),
        {
           'n_estimators': np.arange(5, 10),
           'n_estimators': np.arange(5, 6),
           'min_samples_leaf': [1],
           # 'class_weight': ['balanced']
           # 'max_depth': np.arange(1, 4),
           # 'min_samples_leaf': np.arange(1, 10)
        },
        scoring=f1_macro_scoring,
        cv=cv_count,
        n_jobs=n_jobs)
