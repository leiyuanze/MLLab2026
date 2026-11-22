import numpy as np
import pandas as pd

class WeakClassifier:
    def __init__(self):
        self.tree = None 
        self.alpha = None
    
    def best_split(self, X, y, sample_weight):
        ''' 
        Find the optimal feature and threshold for splitting the data using Gini impurity.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).
            sample_weight (np.ndarray): Sample weights, shape (n_samples,).

        Returns:
            best_feature (str): Name of the feature selected for the split.
            best_threshold (float or int): Optimal threshold for the split.
            best_splits (tuple): Tuple containing masks for left and right splits (np.ndarray, np.ndarray).
        '''
        # TODO: Implement the function to find the best feature, threshold and splits to split the data based on the Gini impurity.
        sample_weight = np.asarray(sample_weight, dtype=float)
        sample_weight = sample_weight / sample_weight.sum()
        y_array = np.asarray(y)

        best_impurity = np.inf
        best_feature = None
        best_threshold = None
        best_splits = None

        def weighted_gini(mask):
            mask = np.asarray(mask, dtype=bool)
            weight = sample_weight[mask].sum()
            if weight == 0:
                return 0.0
            labels, counts = np.unique(y_array[mask], return_counts=True)
            probabilities = counts / counts.sum()
            return weight * (1.0 - np.sum(probabilities ** 2))

        for feature in X.columns:
            values = np.sort(X[feature].unique())
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_mask = (X[feature] <= threshold).to_numpy()
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                impurity = weighted_gini(left_mask) + weighted_gini(right_mask)
                if impurity < best_impurity - 1e-12:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = (left_mask, right_mask)

        if best_feature is None:
            best_feature = X.columns[0]
            best_threshold = X[best_feature].iloc[0]
            left_mask = np.ones(len(X), dtype=bool)
            best_splits = (left_mask, ~left_mask)

        return best_feature, best_threshold, best_splits
        
        
    def fit(self, X, y, sample_weight):
        '''  
        Train the weak classifier on the data.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).
            sample_weight (np.ndarray): Sample weights, shape (n_samples,).

        Returns:
            None: Updates self.tree with the trained decision tree structure.
        '''
        best_feature, best_threshold, best_splits = self.best_split(X, y, sample_weight)
        # TODO: Create the tree as a nested dictionary
        sample_weight = np.asarray(sample_weight, dtype=float)
        y_array = np.asarray(y)
        left_mask, right_mask = best_splits

        def majority_label(mask):
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                return -1
            labels = np.unique(y_array[mask])
            class_weights = [
                sample_weight[mask & (y_array == label)].sum()
                for label in labels
            ]
            return labels[np.argmax(class_weights)]

        self.tree = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": majority_label(left_mask),
            "right": majority_label(right_mask),
        }

    def predict(self, x):
        '''  
        Predict labels for the given data.

        Args:
            x (pd.DataFrame): Data features for prediction, shape (n_samples, n_features).

        Returns:
            predict_labels (np.ndarray): Predicted labels, shape (n_samples,).
        '''

        # Initialize list to store prediction results
        predict_labels = []

        # Predict label for each sample
        for i in range(len(x)):
            sample = x.iloc[i,:]

            # TODO: Predict the label of the sample
            if sample[self.tree["feature"]] <= self.tree["threshold"]:
                predict_labels.append(self.tree["left"])
            else:
                predict_labels.append(self.tree["right"])

        return np.array(predict_labels)
    
    
class Adaboost:
    def __init__(self, n_estimators=10):
        '''
        Initialize the Adaboost classifier.

        Args:
            n_estimators (int): Number of weak classifiers in the ensemble.
        '''
        # Number of weak classifiers to use
        self.n_estimators = n_estimators
        # List to store each weak classifier
        self.clfs = []
    
    # AdaBoost training process
    def fit(self, X, y):
        '''  
        Train the Adaboost ensemble.

        Args:
            X (pd.DataFrame): Data features, shape (n_samples, n_features).
            y (pd.Series): Data labels, shape (n_samples,).

        Returns:
            None: Updates the list of weak classifiers (self.clfs).
        '''
        n_samples, m_features = X.shape
    
        # Initialize weights for each sample
        w = np.ones(n_samples) / n_samples
        self.clfs = []
        y_array = np.asarray(y)

        # Train each weak classifier
        for _ in range(self.n_estimators):
            clf = WeakClassifier()

            # 1. Fit the weak classifier using the current weights
            clf.fit(X, y, w)

            # TODO: 2. Predict the labels of the data using the weak classifier
            predictions = clf.predict(X)

            # TODO: 3. Calculate the error rate by comparing predictions to actual labels
            error = float(np.sum(w[predictions != y_array]) / np.sum(w))
            error = np.clip(error, 1e-12, 1 - 1e-12)

            # TODO: 4. Calculate alpha (the weight of the classifier in the final model)
            alpha = float(0.5 * np.log((1.0 - error) / error))

            # TODO: 5. Update the sample weights to emphasize misclassified samples
            w *= np.exp(-alpha * y_array * predictions)

            # Normalize weights so that they sum to one
            w /= np.sum(w)

            # Store the classifier and its weight
            clf.alpha = alpha
            self.clfs.append(clf)
            

    def predict(self, X):
        '''  
        Predict the label for each sample using the Adaboost ensemble.

        Args:
            X (pd.DataFrame): Data features for prediction, shape (n_samples, n_features).

        Returns:
            predict_labels (np.ndarray): Predicted labels, shape (n_samples,).
        '''

        if not self.clfs:
            raise RuntimeError("The Adaboost model must be fitted before prediction")

        # TODO: 1. Compute the predictions of each weak classifier
        weighted_sum = np.zeros(len(X), dtype=float)
        for clf in self.clfs:
            weighted_sum += clf.alpha * clf.predict(X)

        # TODO: 2. Compute the weighted sum of predictions from all classifiers
        # TODO: 3. Determine the final label based on the sign of the weighted sum (if x>0 return 1, else return -1)
        return np.where(weighted_sum > 0, 1, -1)
