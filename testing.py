from meal import extract_features
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(filename):
    #read file
    dataset = pd.read_csv(filename, usecols = [i for i in range(30)], header = None)
    dataset = dataset.dropna(how='any')

    dataset = extract_features(dataset)
    dataset = StandardScaler().fit_transform(dataset)

    with open("PCA", 'rb') as file:
        pca = pickle.load(file)
    
    dataset = pca.transform(dataset)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                "Naive Bayes", "QDA"]

    for name in names:
        with open(name, 'rb') as file:
            model = pickle.load(file)
        predicted = model.predict(dataset)
        print(name)
        print(predicted)

if __name__ == "__main__":
    main("MealNoMealData/mealData5.csv")