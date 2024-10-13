from sklearn.metrics import confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np


def calculate_f1_score(y_test, y_pred):
    """ 
    Caluclates the f1-score

    y_test: The actual target variable
    y_pred: The predicted target variable 
    """ 
    f1scores  = f1_score(y_test, y_pred, average='micro')
    return f1scores

def create_confusion_matrix(y_test, y_pred, visualize=True): 
    """
    Calculate and visualize the econfusion matrix of the model 
    
    y_test: The actual target variable
    y_pred: The predicted target variable 
    visualize: Indicate whether or not you'd like to visualize the confusion matrix
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    if visualize: 
        # Plot the Confusion Matrix
        plt.figure(figsize=(10,7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        return conf_matrix
    else: 
        return conf_matrix
    return -1

def evaluate_cross_val_score(model_object, X_test_processed, y_test): 
    """
    Caluclate the cross validation score mean for this model 

    model: The model with the approproate parameters (after hypyerparmaeter tuned)
    X: The whole pre-processed dataset with the features 
    y: The whoole datasets target variable 
    """

    scoring = make_scorer(f1_score, average='micro')
    model_scores = cross_val_score(model_object, X_test_processed, y_test, cv=5, scoring=scoring)

    return (model_scores, model_scores.mean(), model_scores.std()) 


def svm_feature_importance(svm_model, preprocessor, X, y_test, visualize=True): 
    ## Preprocess the test dataset
    X_test_processed = preprocessor.transform(X)

    perm_importance = permutation_importance(svm_model, X_test_processed.toarray(), y_test, n_repeats=30, random_state=42)
        
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({'Feature': preprocessor.get_feature_names_out(), 'Importance': perm_importance.importances_mean})

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    
    if visualize: 

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Permutation Feature Importances from SVM')
        plt.show()

        return importance_df
    else: 
        return importance_df





    





