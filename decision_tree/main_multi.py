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


def process(folder_path):
    label_folders = {'person': 1, 'airplane': 2, 'car': 0, 'dog': 3}
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


    dtc = DecisionTreeClassifier(max_depth=10, min_samples_split=7, criterion='entropy')
    dtc.fit(X, y)
    y_guess = dtc.predict(df_test)
    dict={'Sample_name': test_image_ids, 'Output_score': y_guess}
    df=pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_32a.csv"), header = False, index = False)

    #Feature selection and grid search
    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X, y)
    X_new_df = pd.DataFrame(X_new, columns=X.columns[selector.get_support()])
    column_headers = list(X_new_df.columns.values)
    x_val_new = pd.DataFrame()
    for i in column_headers: 
        x_val_new[i] = df_test[i].values

    clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 7, min_samples_split = 7)
    clf.fit(X_new_df, y)
    y_pred = clf.predict(x_val_new)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_32b.csv"), header = False, index = False)

        #cost complexity 
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha = 0.00173)
    clf.fit(X, y)
    y_pred = clf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_32c.csv"), header = False, index = False)

    #Random forest
    rf = RandomForestClassifier(criterion='entropy', max_depth= None, min_samples_split=10, n_estimators=200)
    rf.fit(X, y)
    y_pred = rf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_32d.csv"), header = False, index = False)  

    #XgBoost
    rf = XGBClassifier(max_depth=7, sub_sample=0.6, n_estimators=50)
    rf.fit(X, y)
    y_pred = rf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_32e.csv"), header = False, index = False)  

    rf = XGBClassifier(max_depth=7, sub_sample=0.6, n_estimators=50)
    rf.fit(X, y)
    y_pred = rf.predict(df_test)
    dict = {"Image_id": test_image_ids, "Predictions": y_pred}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(out_path, "test_32h.csv"), header = False, index = False)

    
# if _name_ == "_main_":
#     main(sys.argv[1], sys.argv[2],sys.argv[3])

