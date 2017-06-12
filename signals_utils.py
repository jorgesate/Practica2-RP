import cv2
import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib
from KNNClassifier import *
from ParzenClassifier import *
from GMMsk import *
from SVMClassifier import *


class SignalsUtils(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.best_clf = None
        self.X = 0
        self.y = 0
        self.LDA = LinearDiscriminantAnalysis()
        self.classes = ['No senal', 'Peligro', 'Prohibicion', 'STOP']

    def read_images(self, save=True):

        # Lee cada una de las imagenes de Train, y las adjunta al vector X, junto con su etiqueta en y
        # a continuacion aplica reduccion de dimensionalidad con LDA y lo guarda en formato pickle

        CHANNELS = 3
        no_senales_dir = 'ImgsAlumnos/train/NO_SENALES'
        peligro_dir = 'ImgsAlumnos/train/PELIGRO'
        prohibicion_dir = 'ImgsAlumnos/train/PROHIBICION'
        stop_dir = 'ImgsAlumnos/train/STOP'

        train_ext = ('.jpg', '.ppm')

        y = []
        X = []

        for filename in os.listdir(no_senales_dir):
            if os.path.splitext(filename)[1].lower() in train_ext:

                full_path = os.path.join(no_senales_dir, filename)
                I = cv2.imread(full_path)
                I = cv2.resize(I, (25, 25))

                X.append(np.ravel(I))
                y.append(0)

        for folder in os.listdir(peligro_dir):
            for filename in os.listdir(os.path.join(peligro_dir, folder)):
                if os.path.splitext(filename)[1].lower() in train_ext:

                    full_path = os.path.join(peligro_dir, folder, filename)
                    I = cv2.imread(full_path)
                    I = cv2.resize(I, (25, 25))

                    X.append(np.array(np.ravel(I)))
                    y.append(1)

        for folder in os.listdir(prohibicion_dir):
            for filename in os.listdir(os.path.join(prohibicion_dir, folder)):
                if os.path.splitext(filename)[1].lower() in train_ext:

                    full_path = os.path.join(prohibicion_dir, folder, filename)
                    I = cv2.imread(full_path)
                    I = cv2.resize(I, (25, 25))

                    X.append(np.array(np.ravel(I)))
                    y.append(2)

        for folder in os.listdir(stop_dir):
            for filename in os.listdir(os.path.join(stop_dir, folder)):
                if os.path.splitext(filename)[1].lower() in train_ext:

                    full_path = os.path.join(stop_dir, folder, filename)
                    I = cv2.imread(full_path)
                    I = cv2.resize(I, (25, 25))

                    X.append(np.array(np.ravel(I)))
                    y.append(3)

        Xarray = np.asarray(X)
        self.y = np.array(y)

        print('Lectura finalizada')

        self.X = self.LDA.fit(Xarray, self.y).transform(Xarray)

        if save:
            joblib.dump(self.LDA, './LDA.pkl')

        print('LDA fitted')

        return self.X, self.y

    def test_clfs(self, save=True):

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Naive Bayes", "QDA"]

        sc = []

        for name, clf in zip(names, classifiers):
            clf.fit(self.X, self.y)
            print('%s fitted' % (name))
            scores = cross_val_score(clf, self.X, self.y, cv=10)

            print('{:s} => Accuracy: {:.4f} (+/- {:.2f})'.format(name, scores.mean(), scores.std() * 2))
            sc.append(scores.mean())

        self.best_clf = classifiers[np.argmax(sc)]
        print('Best classifier: {}'.format(names[np.argmax(sc)]))

        self.best_clf.fit(self.X, self.y)

        if save:
            joblib.dump(self.best_clf, './model.pkl')

        return self.best_clf

    def test_clfs2(self, save=True):

        classifiers = [
            GMMClassifierSk(10),
            ParzenClassifier(80),
            KNNClassifier(20),
            SVMClassifier(5)]

        names = ["GMM", "Parzen", "KNN", "SVM"]

        sc = []

        for name, clf in zip(names, classifiers):
            clf.fit(self.X, self.y)
            print('%s fitted' % (name))
            scores = cross_val_score(clf, self.X, self.y, cv=10)

            print('{:s} => Accuracy: {:.4f} (+/- {:.2f})'.format(name, scores.mean(), scores.std() * 2))
            sc.append(scores.mean())

        self.best_clf = classifiers[np.argmax(sc)]
        print('Best classifier: {}'.format(names[np.argmax(sc)]))

        self.best_clf.fit(self.X, self.y)

        if save:
            joblib.dump(self.best_clf, './model.pkl')

        return self.best_clf

    def load_model(self):
        self.best_clf = joblib.load('model.pkl')
        self.LDA = joblib.load('LDA.pkl')
        return self.best_clf

    def predict(self, img):

        img_res = cv2.resize(img, (25, 25))
        img_rav = np.ravel(img_res)
        img_transf = self.LDA.transform(img_rav)

        prediction = self.best_clf.predict(img_transf)
        name = self.classes[prediction[0]]

        return prediction, name
