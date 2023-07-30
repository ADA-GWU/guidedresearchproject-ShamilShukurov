import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
import pickle

def plot_sklearn_roc_curve(y_real, y_pred):
    fpr, tpr, _ = roc_curve(y_real, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0, 1], color = 'g')



def base_lgbm(df_train:pd.DataFrame, 
              df_test:pd.DataFrame, 
              target_col:str,
              dataset_name:str,
              model_name:str="LGBM Baseline",
              message:str="Building LGBM Model..."):
    # Train data
    X_train = df_train.drop(columns=target_col)
    y_train = df_train[target_col]

    # Test data
    X_test = df_test.drop(columns=target_col)
    y_test = df_test[target_col]

    print(message)
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)

    prediction_train = clf.predict(X_train)
    prediction_test = clf.predict(X_test)
    prediction_proba_train=clf.predict_proba(X_train)
    prediction_proba_test=clf.predict_proba(X_test)

    cr_test = classification_report(y_test,prediction_test,zero_division=True, output_dict=True)
    f1_test = cr_test['1']['f1-score']
    acc_test = cr_test['accuracy']
    auc_test = roc_auc_score(y_test, prediction_proba_test[:,1])
    cr_train = classification_report(y_train,prediction_train,zero_division=True, output_dict=True)
    f1_train = cr_train['1']['f1-score']
    acc_train = cr_train['accuracy']
    auc_train = roc_auc_score(y_train, prediction_proba_train[:,1])

    report = {
        "Dataset":dataset_name,
        "Model":model_name,
        "f1_test": f1_test,
        "f1_train":f1_train,
        "accuracy_test" : acc_test,
        "accuracy_train" :acc_train,
        "AUC_test" :auc_test,
        "AUC_train" :auc_train
    }
    print("ROC Curve for test data: {}".format(auc_test))
    plot_sklearn_roc_curve(y_test,prediction_proba_test[:,1])
    
    return clf, report


def upsample(df_train:pd.DataFrame, 
              df_test:pd.DataFrame, 
              target_col:str, 
              dataset_name:str,
              model_name:str="LGBM Upsample",
              message:str="Building LGBM Upsample Model..."):
    print("Upsampling is being applied...")
    df_majority = df_train[(df_train[target_col]==0)] 
    df_minority = df_train[(df_train[target_col]==1)] 
    # upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                  replace=True,    # sample with replacement
                                  n_samples= df_majority.shape[0], # to match majority class
                                  random_state=42)  # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_minority_upsampled, df_majority])
    return base_lgbm(df_upsampled, df_test, target_col, dataset_name, model_name, message)



def downsample(df_train:pd.DataFrame, 
               df_test:pd.DataFrame, 
               target_col:str, 
               dataset_name:str,
               model_name:str="LGBM Downsample",
               message:str="Building LGBM Downsample Model..."):
    print("Downsampling is being applied...")
    df_majority = df_train[(df_train[target_col]==0)] 
    df_minority = df_train[(df_train[target_col]==1)] 
    # downsample majority class
    df_majority_upsampled = resample(df_majority, 
                                  replace=False,    # sample with replacement
                                  n_samples= df_minority.shape[0], # to match majority class
                                  random_state=42)  # reproducible results
    # Combine majority class with upsampled minority class
    df_downsampled = pd.concat([df_minority, df_majority_upsampled])
    return base_lgbm(df_downsampled, df_test, target_col, dataset_name, model_name, message)




def smote_lgbm(df_train:pd.DataFrame, 
               df_test:pd.DataFrame, 
               target_col:str, 
               dataset_name:str,
               model_name:str="SMOTE LGBM",
               message:str="Building SMOTE LGBM Model..."):
    # Train data
    X_train = df_train.drop(columns=target_col)
    y_train = df_train[target_col]
    print("SMOTE is being applied...")
    # Resampling the minority class. The strategy can be changed as required.
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Fit the model to generate the data.
    oversampled_X, oversampled_Y = sm.fit_resample(X_train, y_train)
    oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
    return base_lgbm(oversampled, df_test, target_col, dataset_name, model_name, message)




def balanced_bagging_lgbm(df_train:pd.DataFrame, 
                          df_test:pd.DataFrame, 
                          target_col:str, 
                          dataset_name:str,
                          model_name:str="LGBM Balanced Bagging",
                          message:str="Building LGBM Balanced Bagging Model..."):
    # Train data
    X_train = df_train.drop(columns=target_col)
    y_train = df_train[target_col]

    # Test data
    X_test = df_test.drop(columns=target_col)
    y_test = df_test[target_col]

    print(message)
    clf = lgb.LGBMClassifier()
    classifier = BalancedBaggingClassifier(estimator=clf,
                                  sampling_strategy='not majority',
                                  replacement=False,
                                  random_state=42)
    classifier.fit(X_train, y_train)

    prediction_train = classifier.predict(X_train)
    prediction_test = classifier.predict(X_test)
    prediction_proba_train=classifier.predict_proba(X_train)
    prediction_proba_test=classifier.predict_proba(X_test)

    cr_test = classification_report(y_test,prediction_test,zero_division=True, output_dict=True)
    f1_test = cr_test['1']['f1-score']
    acc_test = cr_test['accuracy']
    auc_test = roc_auc_score(y_test, prediction_proba_test[:,1])
    cr_train = classification_report(y_train,prediction_train,zero_division=True, output_dict=True)
    f1_train = cr_train['1']['f1-score']
    acc_train = cr_train['accuracy']
    auc_train = roc_auc_score(y_train, prediction_proba_train[:,1])

    report = {
      "Dataset":dataset_name,
      "Model":model_name,
      "f1_test": f1_test,
      "f1_train":f1_train,
      "accuracy_test" : acc_test,
      "accuracy_train" :acc_train,
      "AUC_test" :auc_test,
      "AUC_train" :auc_train
    }
    print("ROC Curve for test data: {}".format(auc_test))
    plot_sklearn_roc_curve(y_test,prediction_proba_test[:,1])
    return classifier, report



def lgbm_imbalance(df_train:pd.DataFrame, 
                   df_test:pd.DataFrame, 
                   target_col:str, 
                   dataset_name:str,
                   model_name:str="LGBM_Imbalance",
                   message:str="Building LGBM Imbalance Model..."):
    # Train data
    X_train = df_train.drop(columns=target_col)
    y_train = df_train[target_col]

    # Test data
    X_test = df_test.drop(columns=target_col)
    y_test = df_test[target_col]  

    d_train=lgb.Dataset(X_train, label=y_train)
    d_test=lgb.Dataset(X_test, label=y_test)

    parameters = {
      'objective': 'binary',
      'metric': 'auc',
      'is_unbalance': 'true'
    }

    clf2 = lgb.train(params=parameters, train_set=d_train, valid_sets=d_test, verbose_eval=0)

    prediction_train_proba = clf2.predict(X_train)
    prediction_test_proba = clf2.predict(X_test)

    prediction_train = np.where(prediction_train_proba>0.5,1,0)
    prediction_test = np.where(prediction_test_proba>0.5,1,0)

    cr_test = classification_report(y_test,prediction_test,zero_division=True, output_dict=True)
    f1_test = cr_test['1']['f1-score']
    acc_test = cr_test['accuracy']
    auc_test = roc_auc_score(y_test, prediction_test_proba)
    cr_train = classification_report(y_train,prediction_train,zero_division=True, output_dict=True)
    f1_train = cr_train['1']['f1-score']
    acc_train = cr_train['accuracy']
    auc_train = roc_auc_score(y_train, prediction_train_proba)

    report = {
      "Dataset":dataset_name,
      "Model":model_name,
      "f1_test": f1_test,
      "f1_train":f1_train,
      "accuracy_test" : acc_test,
      "accuracy_train" :acc_train,
      "AUC_test" :auc_test,
      "AUC_train" :auc_train
    }
    print("ROC Curve for test data: {}".format(auc_test))
    plot_sklearn_roc_curve(y_test, prediction_test_proba)
    return clf2, report 


def built_experiment(df_train:pd.DataFrame, 
                     df_test:pd.DataFrame,
                     target_col:str, 
                     dataset_name:str):
    print("Running Experiment...")
    print("Dataset Name: {}\nTarget Column: {}".format(dataset_name, target_col))

    lgbm_base_model, rep_lgbm_base = base_lgbm(df_train.copy(), df_test.copy(),target_col,dataset_name)
    print("Saving model as pickle...")
    with open("{}_lgbm_base_model.pkl".format(dataset_name), "wb") as f:
        pickle.dump(lgbm_base_model, f)

    upsample_model, rep_upsample = upsample(df_train.copy(), df_test.copy(),target_col,dataset_name)
    print("Saving model as pickle...")
    with open("{}_upsample_model.pkl".format(dataset_name), "wb") as f:
        pickle.dump(upsample_model, f)

    downsample_model, rep_downsample = downsample(df_train.copy(), df_test.copy(),target_col,dataset_name)
    print("Saving model as pickle...")
    with open("{}_downsample_model.pkl".format(dataset_name), "wb") as f:
        pickle.dump(downsample_model, f)

    smote_lgbm_model, rep_smote_lgbm = smote_lgbm(df_train.copy(), df_test.copy(),target_col,dataset_name)
    print("Saving model as pickle...")
    with open("{}_smote_lgbm_model.pkl".format(dataset_name), "wb") as f:
        pickle.dump(smote_lgbm_model, f)

    balanced_bagging_lgbm_model, rep_bb = balanced_bagging_lgbm(df_train.copy(), df_test.copy(),target_col,dataset_name)
    print("Saving model as pickle...")
    with open("{}_balanced_bagging_lgbm_model.pkl".format(dataset_name), "wb") as f:
        pickle.dump(balanced_bagging_lgbm_model, f)

    lgbm_imbalance_model, rep_lgbm_imbalance = lgbm_imbalance(df_train.copy(), df_test.copy(),target_col,dataset_name)
    print("Saving model as pickle...")
    with open("{}_lgbm_imbalance_model.pkl".format(dataset_name), "wb") as f:
        pickle.dump(lgbm_imbalance_model, f)

    return pd.DataFrame([rep_lgbm_base,rep_upsample,rep_downsample,rep_smote_lgbm,rep_bb,rep_lgbm_imbalance])




