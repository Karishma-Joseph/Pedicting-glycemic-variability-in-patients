import pandas as pd 
import numpy as np
import pickle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simps
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def clean_data(label):
        dataset = pd.DataFrame()
        for i in range(5):
                df = pd.read_csv("MealNoMealData/"+label+"{}.csv".format(i+1), usecols = [i for i in range(30)], header = None) 
                dataset = dataset.append(df, ignore_index = True)

        dataset = dataset.dropna(how='any') 
        #dataset = dataset.interpolate(method ='linear', limit_direction ='both') 
        return dataset

def extract_features(dataset):
        rms, auc, dt, rr, dc = [], [], [], [], []
        #extract features
        for i in range(len(dataset.index)):
                dc_row=[]
                rms_row = np.sqrt(np.mean(dataset.iloc[i, 0:30]**2))
                rms.append(rms_row)
                auc_row = abs(simps(dataset.iloc[i, 0:30], dx=1))
                auc.append(auc_row)
                sum = 0
                for j in range(29):
                        difference =  abs(dataset.iloc[i,j] - dataset.iloc[i,j+1])
                        sum += difference
                        dc_row.append(difference)
                dt.append(sum)
                rr_row = sum**2 /auc_row
                rr.append(rr_row)
                hist, _ = np.histogram(dc_row, bins=5, range=(0,15), density=True)
                dc.append(list(hist))
        dataset = pd.DataFrame()
        dataset['RMS'] = rms
        dataset['AUC'] = auc
        dataset['DT'] = dt
        dataset['RR'] = rr
        # dataset['DC1'] = [item[0] for item in dc]
        # dataset['DC2'] = [item[1] for item in dc]
        # dataset['DC3'] = [item[2] for item in dc]
        # dataset['DC4'] = [item[3] for item in dc]
        # dataset['DC5'] = [item[4] for item in dc]
        return dataset

def dimensionionality_reduction(k, dataset):
        pca = PCA(n_components=k)
        dataset = pca.fit_transform(dataset)
        with open("PCA", 'wb') as file:
                pickle.dump(pca, file)
        return dataset

def save_model(X, y, names, classifiers):
        for name, classifier in zip(names, classifiers):
                classifier.fit(X, y) 
                with open(name, 'wb') as file:
                        pickle.dump(classifier, file)


def k_fold(X, y, names, classifiers):
        #k-fold cross validation
        kf = KFold(n_splits=10, shuffle=True)
        for name, classifier in zip(names, classifiers):
                accuracy, f1, precision, recall = [], [], [], []
                for train_index, test_index in kf.split(X):
                        Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
                        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

                        clf = classifier

                        clf.fit(Xtrain, ytrain) 
                        predicted = clf.predict(Xtest)

                        accuracy.append(metrics.accuracy_score(ytest, predicted))
                        f1.append(metrics.f1_score(ytest, predicted))
                        precision.append(metrics.precision_score(ytest, predicted))
                        recall.append(metrics.recall_score(ytest, predicted))
                #metrics
                print('Accuracy metrics for ' +name)
                print('Accuracy: ', np.mean(accuracy))
                print('F1 score: ', np.mean(f1))
                print('Precison: ', np.mean(precision))
                print('Recall: ', np.mean(recall))

def main():
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                "Naive Bayes", "QDA"]
        classifiers = [
                KNeighborsClassifier(1),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1, max_iter=1000),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()]
        data1 = clean_data('mealData')
        data2 = clean_data('Nomeal')
        dataset = data1.append(data2, ignore_index = True)

        #extract features
        dataset = extract_features(dataset)

        #scale
        dataset = StandardScaler().fit_transform(dataset)

        #dimensionality reduction
        k=4
        dataset = dimensionionality_reduction(k, dataset)

        dataset = pd.DataFrame(dataset)

        dataset['target'] = [1] * len(data1.index) + [0] * len(data2.index)

        #randomize
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        X = dataset.iloc[:,0:k]
        y = dataset.target

        save_model(X,y, names, classifiers)
        #k-fold
        k_fold(X,y, names, classifiers)

if __name__ == "__main__":
    main()

