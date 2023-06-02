import argparse
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn import metrics
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier

class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature=feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
                
class DecisionTree_info:
        def __init__(self, max_depth=10, min_samples_split=7):
                        self.max_depth = max_depth
                        self.min_samples_split = min_samples_split

        #Fit function essentially needs to construct a tree. 
        def fit(self, X, y):
            self.n_classes_ = len(np.unique(y))
            self.tree_ = self._grow_tree(X, y)
        def _best_split(self, X, y):
                m=y.size
                #No splitting reqd. if only 1 (or less) node
                if m <= 1 or m < self.min_samples_split:
                        return None, None
                #num_parent[c] contains the number of samples in class c
                num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
                #information gain has to be maximized
                best_gain = -1.0
                best_feature, best_threshold = None, None
                
                #entropy of parent node
                entropy_parent = sum(-(num_parent[i]/m)*np.log2(num_parent[i]/m) if num_parent[i] != 0 else 0 for i in range(self.n_classes_))
                
                #Iterate over each feature of X
                for feature in range(X.shape[1]):
                        #Sort in ascending order and unpack in two lists
                        thresholds, classes = zip(*sorted(zip(X.iloc[:, feature], y)))
                        #Creating a list of two elements which will be updates (no. of 0s, no. of 1s)
                        num_left = [0] * self.n_classes_
                        #initially, all the samples are on the right side of the split
                        num_right = num_parent.copy()
                        
                        #We iterate over all the possible splits, calculate information gain, and then decide. No. of splits = no. of samples
                        for i in range(1, m):
                                #classes[] has all the corresponding labels. We are adding the top element to the left node
                                #And we are subtracting that top element from the right one
                                c = classes[i - 1]
                                num_left[c] += 1
                                num_right[c] -= 1
                                entropy_left = sum(-(num_left[x]/i)*np.log2(num_left[x]/i) if num_left[x] != 0 else 0 for x in range(self.n_classes_))
                                entropy_right = sum(-(num_right[x]/(m-i))*np.log2(num_right[x]/(m-i)) if num_right[x] != 0 else 0 for x in range(self.n_classes_))
                                entropy = (i * entropy_left + (m - i) * entropy_right) / m
                                gain = entropy_parent - entropy
                                
                                #If current threshold value is the same as the previous one, we dont do anything and move on to the next threshold value
                                if thresholds[i] == thresholds[i - 1]:
                                    continue

                                if gain > best_gain:
                                        best_gain = gain
                                        best_feature = feature
                                        best_threshold = (thresholds[i] + thresholds[i - 1]) / 2

                #Suppose we iterate over all features, and still best_gain doesnt change
                if best_gain == -1.0:
                    return None, None

                return best_feature, best_threshold

        #Recursively builds the decision tree based on the split returned by best split
        def _grow_tree(self, X, y, depth=0):
                #contains the number of samples that belong to each class in the current node.
                num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
                #predicted_class is the class that is most prevalent in the current node, 
                predicted_class = np.argmax(num_samples_per_class)
                node = Node(value=predicted_class)
                
                if depth < self.max_depth:
                        #Tree has scope to grow more
                        feature, threshold = self._best_split(X, y)
                        if feature is not None:
                                #X[:, feature] < threshold generates a boolean array where 
                                #the value is True if the value of that feature in that sample is less than the threshold, 
                                #and False otherwise. The resulting indices_left array contains True values for the samples that 
                                #belong to the left child, and False values for the samples that belong to the right child.
                                
                                """X[~indices_left]" selects the rows of the NumPy array or 
                                Pandas DataFrame X for which the corresponding boolean value 
                                in the boolean array "indices_left" is False."""
                                
                                indices_left = X.iloc[:, feature] < threshold
                                X_left, y_left = X[indices_left], y[indices_left]
                                X_right, y_right = X[~indices_left], y[~indices_left]
                                
                                node = Node(feature=feature, threshold=threshold)
                                node.left = self._grow_tree(X_left, y_left, depth + 1)
                                node.right = self._grow_tree(X_right, y_right, depth + 1)
                                
                return node
        def predict(self, X):
            X = np.array(X)
            y_pred = []
            for i in range(X.shape[0]):
                y_pred.append(self._predict(X[i]))
            return y_pred
        def _predict(self, inputs):
            node = self.tree_
            while node.left or node.right:
                if inputs[node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.value

def process(folder_path):
    label_folders = {'person': 1, 'airplane': 0, 'car': 0, 'dog': 0}
    data=[]
    #Main folder which contains 4 subfolders
    for folder in os.listdir(folder_path):
        if folder in label_folders:
            label = label_folders[folder]
            folder_images = os.listdir(os.path.join(folder_path, folder))
            for image_name in folder_images:
                if not image_name.startswith('.') and image_name.endswith(('png')):
                    image_path = os.path.join(folder_path, folder, image_name)
                    image = cv2.imread(image_path)
                    if image is not None:
                        data.append([image,label])
                    else:
                        print("Could not read image: " + image_path)
    df = pd.DataFrame()
    for i in range(len(data)):
        row = data[i][0].reshape(1,-1)
        row = row.flatten()
        row = np.append(row,data[i][1])
        df = df.append(pd.Series(row), ignore_index=True)
    X = df.iloc[:,:-1]
    y = df[3072]
    return X,y

def testprocess(folder_path):
    image_size = (32, 32)

    data = []
    test_image_ids = []

    # Loop through all the image files in the folder
    for filename in os.listdir(folder_path):
            # Check if the file is an image
            if filename.endswith(('jpeg', 'png', 'jpg')):
                    # Read the image file
                    image_id = filename.split(".")[0]
                    test_image_ids.append(image_id)
                    img = cv2.imread(os.path.join(folder_path, filename))
                    if img is not None:
                            # Resize the image to 32x32x3
                            resized_img = cv2.resize(img, image_size)
                            # Append the resized image to the data list
                            data.append(resized_img)
                    else:
                        print(f"Could not read image: {filename}")
    df_test = pd.DataFrame()
    for i in range(len(data)):
        row = data[i].reshape(1,-1)
        row = row.flatten()
        df_test = df_test.append(pd.Series(row), ignore_index=True)
    return df_test, test_image_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process training and test data and output a CSV file.')
    parser.add_argument('--train_path', type=str, help='path to train folder')
    parser.add_argument('--test_path', type=str, help='path to test folder')
    parser.add_argument('--out_path', type=str, help='path to store csv file')

    args = parser.parse_args()

        # Accessing the parsed arguments
    train_path = args.train_path
    test_path = args.test_path
    out_path = args.out_path

    X,y=process(train_path)
    df_test, test_image_ids=testprocess(test_path)

    t=DecisionTree_info()
    t.fit(X, y)
    y_guess = t.predict(df_test)

    dict={'Sample_name': test_image_ids, 'Output_score': y_guess}
    df=pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31a.csv"), header = False, index = False)
    dtc = DecisionTreeClassifier(max_depth=10, min_samples_split=7, criterion='entropy')
    dtc.fit(X, y)
    y_guess = dtc.predict(df_test)
    dict={'Sample_name': test_image_ids, 'Output_score': y_guess}
    df=pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31b.csv"), header = False, index = False)

    #Feature selection and grid search
    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X, y)
    X_new_df = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])
    column_headers = list(X_new_df.columns.values)
    x_val_new = pd.DataFrame()
    for i in column_headers: 
        x_val_new[i] = df_test[i].values

    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_split = 4)
    clf.fit(X_new_df, y)
    y_pred = clf.predict(x_val_new)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31c.csv"), header = False, index = False)

        #cost complexity 
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha = 0.0011508)
    clf.fit(X, y)
    y_pred = clf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31d.csv"), header = False, index = False)

    #Random forest
    rf = RandomForestClassifier(criterion='entropy', max_depth= None, min_samples_split=5, n_estimators=100)
    rf.fit(X, y)
    y_pred = rf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31e.csv"), header = False, index = False)  

    #XgBoost
    rf = XGBClassifier(max_depth=6, sub_sample=0.6, n_estimators=30)
    rf.fit(X, y)
    y_pred = rf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31f.csv"), header = False, index = False) 

    rf = XGBClassifier(max_depth=6, sub_sample=0.6, n_estimators=30)
    rf.fit(X, y)
    y_pred = rf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_31h.csv"), header = False, index = False) 


    
     # main(sysargs)

