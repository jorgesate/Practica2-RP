from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


class ParzenClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, rmax, rmin=25, rstep=1):
        self.rmax = rmax
        self.rmin = rmin
        self.rstep = rstep
        self.rs = np.arange(self.rmin, self.rmax + 1, self.rstep)
        self.PARZclsf_best = None
        self.r_best = 0
        self.weights = ['uniform', 'distance']
        self.w_best = None

    def fit(self, X, y):

        scores_matrix = np.zeros([len(self.weights), len(self.rs)])
        for w in self.weights:

            for r in self.rs:
                PARZclsf = RadiusNeighborsClassifier(radius=r, weights=w)
                score_r = cross_val_score(PARZclsf, X, y, cv=5, scoring='f1_macro')

                mean_score = np.mean(score_r)

                xx = np.where(self.rs == r)[0][0]
                yy = self.weights.index(w)

                scores_matrix[yy, xx] = mean_score

        i, j = np.unravel_index(scores_matrix.argmax(), scores_matrix.shape)
        self.r_best = self.rs[j]
        self.w_best = self.weights[i]

        self.PARZclsf_best = RadiusNeighborsClassifier(radius=self.r_best, weights=self.w_best).fit(X, y)

        return self

    def predict(self, X):

        return self.PARZclsf_best.predict(X)
