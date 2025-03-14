import aif360.algorithms.preprocessing
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset, StructuredDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import tensorflow as tf


# Data Loading and Preprocessing
def load_and_preprocess_data():
    # Read the CSV files
    df = pd.read_csv('datasets/CMI_400.csv')
    df_val = pd.read_csv('datasets/CMI_400_val.csv')

    # Display dataset info
    print("Available attributes:", df.columns.tolist())
    print("\nDataset Info:")
    df.info()
    df_val.info()
    
    # Handle missing values
    print("\nNA Values Count:")
    print(df.isna().sum())
    print(df_val.isna().sum())
    
    df_total_na = df.isna().sum().sum()
    print(f"\nTotal number of NA values in training dataset: {df_total_na}")
    df_val_total_na = df_val.isna().sum().sum()
    print(f"\nTotal number of NA values in validation dataset : {df_val_total_na}")
    
    # Fill NA values with mean for numeric columns
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df.fillna({column:df[column].mean()}, inplace=True)
    
    print("\nNA Values Count of training dataset after filling:")
    print(df.isna().sum())
    print(f"\nTotal number of NA values of training dataset after filling: {df.isna().sum().sum()}")

    # Fill NA values with mean for numeric columns
    for column in df_val.columns:
        if df_val[column].dtype in ['int64', 'float64']:
            df_val.fillna({column: df[column].mean()}, inplace=True)

    print("\nNA Values Count of validation dataset after filling:")
    print(df_val.isna().sum())
    print(f"\nTotal number of NA values of validation dataset after filling: {df_val.isna().sum().sum()}")

    # Remove ID column
    df = df.drop('ID', axis=1)
    df_val = df_val.drop('ID', axis=1)
    
    return df,df_val

# Create Binary Label Dataset
def create_binary_dataset(df):
    label_name = ['target']
    protected_attribute_names = ['Sex']
    
    dataset = BinaryLabelDataset(
        df=df,
        label_names=label_name,
        protected_attribute_names=protected_attribute_names,
        favorable_label=1,
        unfavorable_label=0
    )
    
    # Verify dataset properties
    verify_dataset(dataset, protected_attribute_names)
    
    return dataset, label_name, protected_attribute_names

def verify_dataset(dataset, protected_attribute_names):
    unique_labels = dataset.labels.ravel()
    print("\nUnique labels in the dataset:", np.unique(unique_labels))
    
    if len(np.unique(unique_labels)) != 2:
        raise ValueError("Dataset does not contain binary labels")
        
    for attr in protected_attribute_names:
        unique_attrs = dataset.protected_attributes[:, dataset.protected_attribute_names.index(attr)]
        print(f"\nUnique values in protected attribute '{attr}':", np.unique(unique_attrs))
        if len(np.unique(unique_attrs)) != 2:
            raise ValueError(f"Protected attribute '{attr}' is not binary")

# Model Training and Evaluation
def train_and_evaluate_model(dataset_1, dataset_2):
    # Define privileged/unprivileged groups
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    
    # Calculate initial bias metrics
    metrics = BinaryLabelDatasetMetric(
        dataset_1,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    
    print("\nBias metrics on original dataset:")
    print("Mean difference:", metrics.mean_difference())
    print("Disparate impact:", metrics.disparate_impact())
    
    # Split data
    X_train = dataset_1.features
    y_train = dataset_1.labels.ravel()

    # Calculate initial bias metrics
    metrics_val = BinaryLabelDatasetMetric(
        dataset_2,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)

    print("\nBias metrics on validation dataset:")
    print("Mean difference:", metrics_val.mean_difference())
    print("Disparate impact:", metrics_val.disparate_impact())

    # Split data
    X_val = dataset_2.features
    y_val = dataset_2.labels.ravel()

    
    # Train SVM model
    svm_model = SVC(C=10, gamma=0.1, kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = svm_model.predict(X_val)
    print("\nSVM Model Performance on Validation Set:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return X_train, X_val, y_train, y_val, y_pred, privileged_groups, unprivileged_groups

# Bias Mitigation
def apply_reweighting(dataset_Binary, privileged_groups, unprivileged_groups):
    RW = aif360.algorithms.preprocessing.Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Apply reweighting
    RW.fit(dataset_Binary)
    dataset_transf_train = RW.transform(dataset_Binary)

    # dataset_transf_val = RW.transform(dataset_val)
    # print(dataset_transf_val)
    
    # Calculate post-reweighting metrics
    metrics_transf = BinaryLabelDatasetMetric(
        dataset_transf_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    print("\nBias metrics after reweighting training set:")
    print("Mean difference:", metrics_transf.mean_difference())
    print("Disparate impact:", metrics_transf.disparate_impact())
    
    return dataset_transf_train

def apply_disparate_impact_remover(dataset_Binary, privileged_groups, unprivileged_groups):
    # Initialize the Disparate Impact Remover
    DIR = aif360.algorithms.preprocessing.DisparateImpactRemover(
        repair_level=1.0,  # Repair level between 0 and 1
        sensitive_attribute='Sex'
    )
    
    # Apply disparate impact remover transformation
    dataset_transf_train = DIR.fit_transform(dataset_Binary)
    # dataset_transf_val = DIR.fit_transform(dataset_val)
    
    # Calculate post-transformation metrics
    metrics_transf = BinaryLabelDatasetMetric(
        dataset_transf_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    print("\nBias metrics after disparate impact removal in training set:")
    print("Mean difference:", metrics_transf.mean_difference())
    print("Disparate impact:", metrics_transf.disparate_impact())
    
    return dataset_transf_train

def apply_adversarial_debiasing(dataset_Binary, privileged_groups, unprivileged_groups):
    # Initialize the Adversarial Debiasing model
    # tf.compat.v1.reset_default_graph()  # Clears all variables
    tf.compat.v1.disable_eager_execution()  # Ensure TF2 compatibility

    sess = tf.compat.v1.Session()

    debiased_model = aif360.algorithms.inprocessing.AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name='debiased_classifier',
        debias=True,
        sess=sess,
        num_epochs=50,
        batch_size=128,
        classifier_num_hidden_units=128
    )
    
    # Train the model
    debiased_model.fit(dataset_Binary)
    
    # Get predictions on training data
    dataset_debiasing_train = debiased_model.predict(dataset_Binary)

    # Calculate post-transformation metrics
    metrics_debiasing = BinaryLabelDatasetMetric(
        dataset_debiasing_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    print("\nBias metrics after adversarial debiasing on training set:")
    print("Mean difference:", metrics_debiasing.mean_difference())
    print("Disparate impact:", metrics_debiasing.disparate_impact())
    
    return dataset_debiasing_train

def apply_PrejudiceRemover(dataset_Binary, privileged_groups, unprivileged_groups):
    # Initialize the ART Classifier

    PrejudiceRemover = aif360.algorithms.inprocessing.PrejudiceRemover()

    # Train the model
    PrejudiceRemover.fit(dataset_Binary)

    # Get predictions on training data
    dataset_PR_train = PrejudiceRemover.predict(dataset_Binary)

    # Calculate post-transformation metrics
    metrics_PR = BinaryLabelDatasetMetric(
        dataset_PR_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    print("\nBias metrics after Predujice Remover Technique on training set:")
    print("Mean difference:", metrics_PR.mean_difference())
    print("Disparate impact:", metrics_PR.disparate_impact())

    return dataset_PR_train

def apply_SMOTE(dataset_Binary, dataset_val, privileged_groups, unprivileged_groups):

    smote = SMOTE(random_state=2)

    X_train = dataset_Binary.features
    y_train = dataset_Binary.labels.ravel()

    X_train_transf,y_train_transf  = smote.fit_resample(X_train, y_train)
    X_val = dataset_val.features
    y_val = dataset_val.labels.ravel()

    # scaling of data
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train_transf)
    X_train_scaled = scaler.transform(X_train_transf)
    X_val_scaled = scaler.transform(X_val)

    # Train SVM model
    svm_model = SVC(C=10, gamma=0.1, kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train_transf)

    # Evaluate model
    y_pred = svm_model.predict(X_val_scaled)
    print("\nSVM Model Performance on Validation Set (SMOTE):")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report (SMOTE):")
    print(classification_report(y_val, y_pred))



    # scaling of data
    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(X_train_transf)
    # X_train_scaled = scaler.transform(X_train_transf)

    # # Convert NumPy arrays to pandas DataFrame/Series for proper concatenation
    # X_train_transf = pd.DataFrame(X_train_transf, columns=dataset_Binary.feature_names)
    # y_train_transf_series = pd.Series(y_train_transf, name='target')
    # # Combine the scaled features and transformed target into a single DataFrame
    # dataset_transf_train_df = pd.concat([X_train_transf, y_train_transf_series], axis=1)
    #
    # print(dataset_transf_train_df.head(5))


    # # Convert target column to numeric type if it's not already
    # dataset_transf_train_df['target'] = pd.to_numeric(dataset_transf_train_df['target'])
    #
    # # dataset_transf_SMOTE = create_binary_dataset(dataset_transf_train_df)
    # #
    # # metrics_transf_SMOTE = BinaryLabelDatasetMetric(
    # #     dataset_transf_SMOTE,
    # #     unprivileged_groups=unprivileged_groups,
    # #     privileged_groups=privileged_groups
    # # )
    # #
    # # print("\nBias metrics after SMOTE in training set:")
    # # print("Mean difference:", metrics_transf_SMOTE.mean_difference())
    # # print("Disparate impact:", metrics_transf_SMOTE.disparate_impact())

    return
# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df, df_val= load_and_preprocess_data()
    
    # Create binary dataset
    dataset_Binary, label_name, protected_attribute_names = create_binary_dataset(df)
    dataset_val, label_name, protected_attribute_names = create_binary_dataset(df_val)
    
    # Train and evaluate initial model
    X_train, X_val, y_train, y_val, y_pred, privileged_groups, unprivileged_groups = train_and_evaluate_model(dataset_Binary, dataset_val)
    
    # # Create validation dataset
    # val_df = pd.DataFrame(X_val, columns=dataset_Binary.feature_names)
    # val_df['target'] = y_val
    # val_df['Sex'] = X_val[:, dataset_Binary.protected_attribute_names.index('Sex')]
    #
    #
    # dataset_val_pred = dataset_val.copy()
    # dataset_val_pred.labels = y_pred.reshape(-1, 1)
    
    # # Apply bias mitigation reweighting
    # dataset_transf_train_RW = apply_reweighting(
    #     dataset_Binary,
    #     privileged_groups,
    #     unprivileged_groups
    # )
    # # Apply DIR bias mitigation
    # dataset_transf_train_DIR = apply_disparate_impact_remover(
    #     dataset_Binary,
    #     privileged_groups,
    #     unprivileged_groups
    # )
    # # Apply AdDebias inprocessing.
    # dataset_transf_train_AdDe = apply_adversarial_debiasing(
    #     dataset_Binary,
    #     privileged_groups,
    #     unprivileged_groups)

    # Apply AdDebias inprocessing.
    dataset_transf_train_PR = apply_PrejudiceRemover(
        dataset_Binary,
        privileged_groups,
        unprivileged_groups)

    # print('SMOTE-Vanilla')
    # apply_SMOTE(dataset_Binary, dataset_val, privileged_groups, unprivileged_groups)
    # print('SMOTE-REWEIGHTED')
    # apply_SMOTE(dataset_transf_train_RW,dataset_val, privileged_groups, unprivileged_groups)
    # print('SMOTE-DIR')
    # apply_SMOTE(dataset_transf_train_DIR,dataset_val, privileged_groups, unprivileged_groups)
    print("SMOTE-Prejudice Remover")
    apply_SMOTE(dataset_transf_train_PR, dataset_val, privileged_groups, unprivileged_groups)

    # Train and evaluate transformed model
    # X_train_transf, X_val_transf, y_train_transf, y_val_transf, y_pred_transf, privileged_groups, unprivileged_groups = train_and_evaluate_model(
    #     dataset_transf_train_RW, dataset_val)
    #
    #
    # # X_train_transf, X_val_transf, y_train_transf, y_val_transf, y_pred_transf, privileged_groups, unprivileged_groups = train_and_evaluate_model(
    # #     dataset_transf_SMOTE, dataset_val)
    #
    #
    # print("\nFinal SVM Model Performance:")
    # print("Accuracy:", accuracy_score(y_val_transf, y_pred_transf))
    # print("\nClassification Report:")
    # print(classification_report(y_val_transf, y_pred_transf))
    
