from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
import numpy as np


class GMMClassifierSk(BaseEstimator, ClassifierMixin):

    def __init__(self, nmax, nmin=2, nstep=1):
        self.nmax = nmax
        self.nmin = nmin
        self.nstep = nstep
        self.ns = np.arange(self.nmin, self.nmax + 1, self.nstep)
        self.GMMclsf_best = []
        self.n_best = []
        self.covs_type = ['full', 'tied', 'diag', 'spherical']
        self.cov_best = []
        self.n_classes = 0

    def fit(self, X, y):

        self.n_classes = len(np.unique(y))
        for nc in range(self.n_classes):

            scores_matrix = np.zeros([len(self.covs_type), len(self.ns)])
            for ct in self.covs_type:

                for n in self.ns:
                    GMMclsf = GaussianMixture(n_components=n, covariance_type=ct)
                    score_k = cross_val_score(GMMclsf, X, y, cv=5, scoring='f1_macro')
                    mean_score = np.mean(score_k)

                    xx = np.where(self.ns == n)[0][0]
                    yy = self.covs_type.index(ct)

                    scores_matrix[yy, xx] = mean_score

            i, j = np.unravel_index(scores_matrix.argmax(), scores_matrix.shape)
            self.n_best.append(self.ns[j])
            self.cov_best.append(self.covs_type[i])

            self.GMMclsf_best.append(GaussianMixture(n_components=self.n_best[nc], covariance_type=self.cov_best[nc])
                                     .fit(X[y == nc]))

        return self

    def predict(self, X):

        n_samples = X.shape[0]
        predicts = np.zeros((self.n_classes, n_samples))
        for n in range(self.n_classes):
            predicts[n, :] = self.GMMclsf_best[n].score_samples(X)
        return np.argmax(predicts, axis=0)




