import os
import joblib
import xgboost
import argparse
import numpy as np
import pandas as pd
from azureml.core.run import Run
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    # Clean and one hot encode data
    # First remove the question mark?
    data.columns = data.columns.str.replace(r'?', '')
    # shorten column names    
    data.columns = ['Age', 'Gender', 'T_Bili', 'D_Bili', 'AA_Phosphate', 'SgptA_Aminotrans', 'SgotA_Aminotrans','T_proteins', 'ALB_Albumin', 'AG_AlbuminR_GlobulinR','Result']
    # Remove null rows since it is just a bit above 10% of the the entire dataset.
    data_new = data.dropna(how='any',axis=0)
    # one hot encoding using pandas method (get_dummies)
    Gen = pd.get_dummies(data_new.Gender, prefix='Gender')
    data_new.drop("Gender", inplace=True, axis=1)
    data_new = data_new.join(Gen)
    X_train = data_new[[x for x in data_new.columns if x not in ["Result"]]]
    X_train = X_train[['Age', 'Gender_Female', 'Gender_Male', 'T_Bili', 'D_Bili', 'AA_Phosphate', 'SgptA_Aminotrans', 'SgotA_Aminotrans','T_proteins', 'ALB_Albumin', 'AG_AlbuminR_GlobulinR']]
    y_train = data_new[["Result"]]
    return X_train, y_train

run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees you want the algorithm to build. The more The more rows in the data, the more trees are needed")
    parser.add_argument('--max_depth', type=int, default=6, help="Controls how specialized each tree is to the training dataset. The more the value the more likely overfitting")
    parser.add_argument('--subsample', type=float, default=1, help="Randomly selected subset of the training dataset to fit each tree. Fewer samples cause more variance for each tree")

    args = parser.parse_args()
    
    run.log("Number of Trees:", np.int(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))
    run.log("Subsample of Dataset:", np.float(args.subsample))

     
    # Data is located at:
    # "https://raw.githubusercontent.com/chollette/Liver-Disease-Classification-Azure-ML-Capstone-Project/master/starter_file/data/Liver%20Patient%20Dataset%20(LPD)_train.csv"
    data_path = "https://raw.githubusercontent.com/chollette/Liver-Disease-Classification-Azure-ML-Capstone-Project/master/starter_file/data/Liver%20Patient%20Dataset%20(LPD)_train.csv"
    df = pd.read_csv(data_path)

    x, y = clean_data(df) #.to_numpy()  

    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    # Train data using xgboost classifier 
    model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, subsample=args.subsample, grow_policy='lossguide', learning_rate =0.1,max_bin=63,max_leaves=63, min_child_weight=1,tree_method='hist',reg_lambda=2.1875, reg_alpha=1.0416666666666667, gamma=0, colsample_bytree=0.8, objective= 'reg:logistic', nthread=None, scale_pos_weight=1, seed=None).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))

    #Save model as joblib.dump(model, "/outputs/model.joblib")
    os.makedirs('./outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(model,'./outputs/model.joblib')


if __name__ == '__main__':
    main()