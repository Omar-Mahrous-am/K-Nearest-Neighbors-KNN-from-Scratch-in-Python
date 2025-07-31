import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    """
    Calculate the Euclidean distance between two points x1 and x2.

    Parameters:
    - x1 (ndarray): First point.
    - x2 (ndarray): Second point.

    Returns:
    - float: The Euclidean distance.
    """
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    def __init__(self, n_neighbors):
        """
        Initialize the KNN classifier.

        Parameters:
        - n_neighbors (int): Number of nearest neighbors to consider.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Store the training data.

        Parameters:
        - X (ndarray): Training features (samples × features).
        - y (ndarray): Training labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        """
        Predict the class labels for the input samples.

        Parameters:
        - x (ndarray): Input data to classify.

        Returns:
        - list: Predicted class labels.
        """
        predictions = [self._predict(i) for i in x]
        return predictions

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Parameters:
        - x (ndarray): A single input sample.

        Returns:
        - int/str/float: Predicted class label.
        """
        distances = [euclidian_distance(x, x_train_sample) for x_train_sample in self.X_train]
        sorted_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearst_neighbors = [self.y_train[sorted_index] for sorted_index in sorted_indices]
        predicted_classes = Counter(k_nearst_neighbors).most_common()
        return predicted_classes[0][0]


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score,f1_score


    X=load_iris().data
    y=load_iris().target
    print(X)
    print(y)

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.20,random_state=42)

    knn_model=KNN(n_neighbors=3)

    knn_model.fit(X_train,y_train)


    y_pred=knn_model.predict(X_test)


    print(f"The Accuracy Score on Test set is {accuracy_score(y_test,y_pred)}")


    print(f"The f1 Score on Test set is {f1_score(y_test,y_pred,average='macro')}")


    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()






    

