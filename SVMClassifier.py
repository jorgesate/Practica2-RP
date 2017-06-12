from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# KNeighborsClassifier(n_neighbors=5, weights='uniform''distance', algorithm='auto',
#       leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)


class SVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, cmax, cmin=1, cstep=0.1):
        self.cmax = cmax
        self.cmin = cmin
        self.cstep = cstep
        self.cs = np.arange(self.cmin, self.cmax + 1, self.cstep)
        self.SVMclsf_best = None
        self.c_best = 0
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        self.kernel_best = None

    def fit(self, X, y):

        scores_matrix = np.zeros([len(self.weights), len(self.ks)])
        for kernel in self.kernels:
            for c in self.cs:
                SVC
                SVMclsf = SVC(C=c, kernel=kernel)
                score_k = cross_val_score(SVMclsf, X, y, cv=5, scoring='f1_macro')
                mean_score = np.mean(score_k)

                xx = np.where(self.cs == c)[0][0]
                yy = self.kernels.index(kernel)

                scores_matrix[yy, xx] = mean_score

        i, j = np.unravel_index(scores_matrix.argmax(), scores_matrix.shape)
        self.c_best = self.cs[j]
        self.kernel_best = self.kernels[i]

        self.SVMclsf_best = SVC(C=self.c_best, kernel=self.kernel_best).fit(X, y)

        return self

    def predict(self, X):

        return self.SVMclsf_best.predict(X)



