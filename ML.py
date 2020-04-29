import pandas as pd
import numpy as np
from subprocess import call
from scipy import stats
from multiprocessing import Pool
from functools import partial
from itertools import chain, combinations, product
from tqdm import tqdm
import seaborn as sb
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
plt.style.use('ggplot')
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.base import clone as skclone
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, make_scorer
from sklearn.feature_selection import RFECV, RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Estimator():
    
    def fit(self, model, X, y):
        self._X        = X
        self._y        = y
        self._features = X.columns
        self._target   = y.columns
        self._model    = model.fit(self.X(), self.y())
        return self

    def data(self):
        data = pd.DataFrame(self.X(), columns=self.features())
        data[self.target()[0]] = self.y()
        return data
    
    def X(self):
        return self._X.values

    def y(self):
        return self._y.values[:,0]

    def model(self):
        return self._model

    def features(self):
        return self._features

    def target(self):
        return self._target

    def feature_importance(self):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(self.X(), self.y(), random_state=42)
        clf.fit(X_train, y_train)
        
        result = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()
        
        tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
        tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, len(self.features())*0.4))
        ax1.barh(tree_indices, clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
        ax1.set_yticklabels(self.features()[tree_importance_sorted_idx])
        ax1.set_yticks(tree_indices)
        ax1.set_ylim((0, len(clf.feature_importances_)))
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=self.features()[perm_sorted_idx])
        fig.tight_layout()
        plt.show()
        
    def feature_to_target_correlation(self, method='spearman'):
        cols = self.features().to_list() + self.target().to_list()
        data = self.data()[cols]
        corr = data.corr(method=method)
        corr = corr[self.target().to_list()]
        corr = corr.sort_values(self.target().to_list(),ascending=False).drop(self.target().to_list())
        fig, ax = plt.subplots(figsize=(1,len(cols)*0.4))
        sb.heatmap(corr,yticklabels=True, ax=ax, annot=True, annot_kws={"size":8})

    def feature_to_feature_correlation(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        corr = spearmanr(self.X()).correlation
        corr_linkage = hierarchy.ward(corr)
        dendro = hierarchy.dendrogram(corr_linkage, labels=self.features(), ax=ax1, leaf_rotation=90)
        dendro_idx = np.arange(0, len(dendro['ivl']))
        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
        ax2.set_yticklabels(dendro['ivl'])
        fig.tight_layout()
        plt.show()
        
    def feature_to_feature_correlation_table(self, method='spearman', show=100):
        cols = self.features().to_list()
        data = self.data()[cols]
        corr = data.corr(method=method)
        fig, ax = plt.subplots(figsize=(len(cols), len(cols)))
        sb.heatmap(corr,yticklabels=True, ax=ax, annot=True, square=True, annot_kws={"size":12})
        
    def score(self, scoring='accuracy', cv=5, jobs=4):
        return cross_val_score(self.model(), self.X(), self.y(), scoring=scoring, cv=cv, n_jobs=jobs).mean()

    def tune(self, parameters, scoring='accuracy', iterations=1000, cv=3, verbose=1):
        results = RandomizedSearchCV(self.model(), parameters, random_state=0, scoring=scoring, cv=cv, refit=False, n_iter=iterations, n_jobs=2, iid=False, verbose=verbose, error_score=0.0)
        results.fit(self.X(), self.y())
        return results

    def recursive_feature_elimination_cv(self, number_of_features_to_select=10, cv=10, scoring='accuracy', steps=1, verbose=1):
        selector = RFECV(self.model(), min_features_to_select=number_of_features_to_select, step=steps, cv=cv, scoring=scoring, verbose=verbose, n_jobs=2)
        selector = selector.fit(self.X(), self.y())
        return selector

    def recursive_feature_elimination(self, number_of_features_to_select=10, step=1):
        selector = RFE(self.model(), n_features_to_select=number_of_features_to_select, step=step)
        selector = selector.fit(self.X(), self.y())
        return (self.features()[selector.support_], selector)

    def accuracy_matrix(self, cv=10, jobs=4):
        X=self.X()
        y=self.y()
        y_pred = cross_val_predict(self.model(), X, y, cv=cv, n_jobs=jobs)
        cm = confusion_matrix(y, y_pred)
        return pd.DataFrame(cm, columns=self.model().classes_, index=self.model().classes_)
    
    def estimator_roc_curve(self, cv=10):
        y_prob   = cross_val_predict(self.model(), X=self.X(), y=self.y(), cv=cv, method='predict_proba')[:,1]
        y_actual = self.y()
        data     = np.concatenate((y_actual.reshape(-1,1), y_prob.reshape(-1,1)), axis=1)
        positive = data[data[:,0]==1][:,1]
        negative = data[data[:,0]==0][:,1]

        x = np.linspace(0, 1, num=100)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        sb.kdeplot(positive, shade=True, ax=ax1, color="g")
        sb.kdeplot(negative, shade=True, ax=ax1, color="r")

        fpr, tpr, thresholds = roc_curve(y_actual, y_prob)
        ax2.plot(fpr, tpr, color='red', label='ROC')
        ax2.plot([0, 1], [0, 1], color='black', linestyle='--')
    
    def feature_selection(self, minimum=5, maximum=None, accept_score=1.0, scoring='accuracy', cv=10, trim_rate=1):
        trim_rate        *= -1
        feature_count    = maximum or len(self.features())# if not maximum else np.min([ maximum, len(self.features()) ])
        features_names   = np.array(self.features())
        features_indexes = np.arange(feature_count)
        best_score       = self.score(scoring=scoring, cv=cv)
        start_best_score = best_score
        results          = []
        final_results    = []
        for size in range(feature_count, minimum,trim_rate):
            size = np.min([ size, len(features_indexes)+trim_rate ])
            subsets = [list(subset) for subset in combinations(features_indexes, size)]
            results = []
            for subset in subsets:
                model = skclone(self.model())
                X     = self.data()[features_names[subset].tolist()]
                y     = self.y()
                score = cross_val_score(model, X, y, scoring=scoring, cv=cv).mean()
                if  score >= best_score:
                    best_score = np.max([best_score, score])
                    results.append({ 'score':score, 'features': subset })
            if  results:
                _results         = pd.DataFrame(results)
                best_score       = _results.score.max()
                features_indexes = np.unique(np.concatenate(_results[_results.score==best_score].features.values))
            elif not results or best_score >= accept_score:
                return features_names[features_indexes].tolist(), best_score
