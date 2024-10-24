import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import os

def loadTrainingSet(datasetFile):
    # Load the entire training dataset to fit the OneHotEncoder
    full_train_dataset = pd.read_csv(datasetFile, delimiter=",", dtype=str, header=None)

    # Identify categorical columns
    categorical_indices = [i for i in range(full_train_dataset.shape[1] - 1) if not full_train_dataset.iloc[0, i].isnumeric()]
    categorical_columns = full_train_dataset.columns[categorical_indices]

    # Initialize and fit OneHotEncoder on the entire dataset
    encoder = OneHotEncoder(sparse_output=True)
    encoder.fit(full_train_dataset.iloc[:, categorical_indices])

    # Load the training dataset in chunks
    chunk_size = 10000  # Adjust based on your memory capacity
    train_dataset_iterator = pd.read_csv(datasetFile, delimiter=",", dtype=str, chunksize=chunk_size, header=None)

    # Process the training dataset in chunks
    training_samples_list = []
    training_labels_list = []

    for chunk in train_dataset_iterator:
        training_samples = chunk.iloc[:, :-1]  # All columns except the last one
        training_labels = chunk.iloc[:, -1].values  # Last column

        # Convert all discrete values to one-hot encoding using the fitted encoder
        categorical_data = encoder.transform(training_samples.iloc[:, categorical_indices])

        # Remove categorical columns from original data
        numerical_indices = [i for i in range(training_samples.shape[1]) if i not in categorical_indices]
        numerical_data = csr_matrix(training_samples.iloc[:, numerical_indices].astype(float))

        # Concatenate numerical and encoded categorical data
        training_samples_sparse = hstack((numerical_data, categorical_data))

        # Append to lists
        training_samples_list.append(training_samples_sparse)
        training_labels_list.append(training_labels)

    # Combine all chunks into a single sparse matrix
    training_samples_sparse = vstack(training_samples_list)
    training_labels = np.concatenate(training_labels_list)

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(training_samples_sparse, training_labels, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val

def trainModel(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

# Generates a pdf report of the training results
def generateTrainingReport(clf, X_val, y_val):
    # Make predictions on the validation set
    y_pred = clf.predict(X_val)
    
    all_classes = sorted(set(y_val) | set(y_pred))
    
    # Generate classification report
    report = classification_report(y_val, y_pred, labels=all_classes, zero_division=1)
    print(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=all_classes)
    plt.figure(figsize=(15, 10))  # Increased figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Generate and save precision-recall curve for each class
    classes = np.unique(y_val)
    y_val_bin = label_binarize(y_val, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    precision_recall_generated = False
    plt.figure()
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_val_bin[:, i], y_pred_bin[:, i])
        average_precision = average_precision_score(y_val_bin[:, i], y_pred_bin[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {classes[i]} (AP = {average_precision:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Multiclass Data')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # Place legend outside the plot
    plt.savefig('precision_recall_curve_multiclass.png', bbox_inches='tight')
    plt.close()
    precision_recall_generated = True

    # Generate and save ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiclass Data')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # Place legend outside the plot
    plt.savefig('roc_curve_multiclass.png', bbox_inches='tight')
    plt.close()
    
    # Create a PDF report
    pdf = FPDF()
    pdf.add_page()
    # Add classification report to the PDF
    pdf.set_font("Courier", size=11)
    for line in report.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    # Add confusion matrix image to the PDF
    pdf.add_page()
    pdf.image("confusion_matrix.png", x=10, y=10, w=190)
    # Add precision-recall curve image to the PDF if it was generated
    if precision_recall_generated:
        pdf.add_page()
        pdf.image("precision_recall_curve_multiclass.png", x=10, y=10, w=190)
    # Add ROC curve image to the PDF
    pdf.add_page()
    pdf.image("roc_curve_multiclass.png", x=10, y=10, w=190)
    # Save the PDF
    pdf.output("classification_report.pdf")
    
    # Remove the generated .png files
    os.remove("confusion_matrix.png")
    if precision_recall_generated:
        os.remove("precision_recall_curve_multiclass.png")
    os.remove("roc_curve_multiclass.png")

# Saves the model to a file
def saveModel(model):
    with open('model.pkl', 'wb') as modelFile:
        pickle.dump(model, modelFile)

# Loads a preexisting model from the specified file
def loadModel(modelFile):
    with open('model.pkl', 'rb') as modelFile:
        return pickle.load(modelFile)
    
def main():
    datasetFile = "data/kddcup.data_10_percent.gz"
    X_train, y_train, X_val, y_val = loadTrainingSet(datasetFile)
    
    debug = False
    if debug:
        # DEBUG: Print the frequency of each class in the training set
        print("Class distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
        
        # DEBUG: Print the frequency of each class in the validation set
        print("Class distribution in validation set:")
        unique, counts = np.unique(y_val, return_counts=True)
        print(dict(zip(unique, counts)))

        
    else:
        # Ask the user if they would like to load a preexisting model or train a new one
        inp = input('Would you like to load a preexisting model? (Y/N)\n>> ').capitalize()
        while inp != 'Y' and inp != 'N':
            print("Invalid input. Please answer with Y or N.")
            inp = input("Would you like to load a preexisting model? (Y/N)\n>> ").capitalize()
            
        if inp == 'Y':
            # Use a preexisting model for classification
            clf = loadModel('model.pkl')
            generateTrainingReport(clf, X_val, y_val)
            
        elif inp == 'N':
            # Train a new model and generate a report
            print("Training a new model with data from " + datasetFile)
            clf = trainModel(X_train, y_train)
            
            # Ask the user if they would like to generate a pdf report
            inp = input('Would you like to generate a pdf report? (Y/N)\n>> ').capitalize()
            while inp != 'Y' and inp != 'N':
                print("Invalid input. Please answer with Y or N.")
                inp = input("Would you like to generate a pdf report? (Y/N)\n>> ").capitalize() 
                        
            if inp == 'Y':
                generateTrainingReport(clf, X_val, y_val)
            elif inp == 'N':
                print("Report not saved.")
                
            # Save the model
            saveModel(clf)
            print("Model saved to model.pkl")
        
main()