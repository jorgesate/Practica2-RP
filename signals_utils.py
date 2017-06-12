import cv2
import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib
from sklearn import metrics
import pylab as plt


class SignalsUtils(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.best_clf = None
        self.X = 0
        self.y = 0
        self.LDA = LinearDiscriminantAnalysis()
        self.PCA = PCA()
        self.classes = ['No senal', 'Peligro', 'Prohibicion', 'STOP']

    def read_images(self, save=True):

        # Lee cada una de las imagenes de Train, y las adjunta al vector X, junto con su etiqueta en y
        # a continuacion aplica reduccion de dimensionalidad con LDA y PCA y lo guarda en formato pickle

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

        Xarray = self.PCA.fit_transform(Xarray, self.y)
        self.X = self.LDA.fit_transform(Xarray, self.y)

        if save:
            joblib.dump(self.LDA, './LDA.pkl')
            joblib.dump(self.PCA, './PCA.pkl')

        print('LDA + PCA fitted')

        return self.X, self.y

    def test_clfs(self, save=True):
        # Realiza un test con varios clasificadores y elije el mejor, ademas lo guarda en formato pickle

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=1),
            SVC(gamma=2, C=1),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Naive Bayes", "QDA"]

        sc = []

        for name, clf in zip(names, classifiers):
            clf.fit(self.X, self.y)
            print('{:s} fitted'.format(name))
            scores = cross_val_score(clf, self.X, self.y, cv=10)

            print('{:s} -> Accuracy: {:.4f} (+/- {:.2f})'.format(name, scores.mean(), scores.std() * 2))
            sc.append(scores.mean())

            # Dibuja la matriz de confusion
            predicted = clf.predict(self.X)
            conf_matrix = metrics.confusion_matrix(self.y, predicted)
            fig = plt.figure(4, figsize=(7, 7), dpi=200)
            mytitle = name + ' classifier'
            fig.canvas.set_window_title(mytitle)
            self.plot_confusion_matrix(conf_matrix, cmap=plt.cm.get_cmap('jet'))
            fig.show()
            # save_fig = './confusion_matrix_figs/' + mytitle
            # fig.savefig(save_fig)
            fig.clf()

        self.best_clf = classifiers[np.argmax(sc)]
        print('Best classifier: {}'.format(names[np.argmax(sc)]))

        self.best_clf.fit(self.X, self.y)

        if save:
            joblib.dump(self.best_clf, './model.pkl')

        return self.best_clf

    def load_model(self):
        # Carga los clasificadores previamente entrenados
        self.best_clf = joblib.load('model.pkl')
        self.LDA = joblib.load('LDA.pkl')
        self.PCA = joblib.load('PCA.pkl')
        return self.best_clf

    def predict(self, img):
        # Realiza una prediccion

        img_res = cv2.resize(img, (25, 25))
        img_rav = np.ravel(img_res)
        img_transf = self.PCA.transform(img_rav)
        img_transf = self.LDA.transform(img_transf)

        prediction = self.best_clf.predict(img_transf)
        name = self.classes[prediction[0]]

        return prediction, name

    # Funcion creada por: Jose' Miguel Buenaposada (josemiguel.buenaposada@urjc.es)
    # para dibujar matrices de confusion
    def plot_confusion_matrix(self, cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        #    plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, range(cm.shape[0]))
        plt.yticks(tick_marks, range(cm.shape[0]))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        ax = plt.gca()
        width = cm.shape[1]
        height = cm.shape[0]

        for x in range(width):
            for y in range(height):
                ax.annotate(str(cm[y, x]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

