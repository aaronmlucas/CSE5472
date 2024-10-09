import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from fpdf import FPDF

# Load the entire training dataset to fit the OneHotEncoder
full_train_dataset = pd.read_csv('data/kddcup.data_10_percent.gz', delimiter=",", dtype=str)

# Identify categorical columns
categorical_indices = [i for i in range(full_train_dataset.shape[1] - 1) if not full_train_dataset.iloc[0, i].isnumeric()]

# Initialize and fit OneHotEncoder on the entire dataset
encoder = OneHotEncoder(sparse_output=True)
encoder.fit(full_train_dataset.iloc[:, categorical_indices])

# Load the training dataset in chunks
chunk_size = 10000  # Adjust based on your memory capacity
train_dataset_iterator = pd.read_csv('data/kddcup.data_10_percent.gz', delimiter=",", dtype=str, chunksize=chunk_size)

# Process the training dataset in chunks
training_samples_list = []
training_labels_list = []

for chunk in train_dataset_iterator:
    training_samples = chunk.iloc[:, :-1].values  # All columns except the last one
    training_labels = chunk.iloc[:, -1].values  # Last column

    # Convert all discrete values to one-hot encoding using the fitted encoder
    categorical_data = encoder.transform(training_samples[:, categorical_indices])

    # Remove categorical columns from original data
    numerical_indices = [i for i in range(training_samples.shape[1]) if i not in categorical_indices]
    numerical_data = csr_matrix(training_samples[:, numerical_indices].astype(float))

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

# Convert sparse matrices to dense format for training (if necessary)
X_train_dense = X_train.toarray()
X_val_dense = X_val.toarray()

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_dense, y_train)

# Make predictions on the validation set
val_predictions = clf.predict(X_val_dense)

# Evaluate the model
report = classification_report(y_val, val_predictions)
print(report)

# Save the classification report to a text file
with open("classification_report.txt", "w") as f:
    f.write(report)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_val, val_predictions)

# Binarize the labels for multi-class precision-recall and ROC curves
y_val_binarized = label_binarize(y_val, classes=clf.classes_)
n_classes = y_val_binarized.shape[1]

# Compute precision-recall pairs for each class
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_val_binarized[:, i], clf.predict_proba(X_val_dense)[:, i])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], clf.predict_proba(X_val_dense)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot and save the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.close()

# Plot and save the precision-recall curve for each class
plt.figure(figsize=(12, 10))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {clf.classes_[i]}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.savefig("precision_recall_curve.png")
plt.close()

# Plot and save the ROC curve for each class
plt.figure(figsize=(12, 10))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {clf.classes_[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig("roc_curve.png")
plt.close()

# Create a PDF report
pdf = FPDF()
pdf.add_page()

# Add classification report to the PDF
pdf.set_font("Arial", size=12)
with open("classification_report.txt", "r") as f:
    for line in f:
        pdf.cell(200, 10, txt=line, ln=True)

# Add confusion matrix image to the PDF
pdf.add_page()
pdf.image("confusion_matrix.png", x=10, y=10, w=190)

# Add precision-recall curve image to the PDF
pdf.add_page()
pdf.image("precision_recall_curve.png", x=10, y=10, w=190)

# Add ROC curve image to the PDF
pdf.add_page()
pdf.image("roc_curve.png", x=10, y=10, w=190)

# Save the PDF
pdf.output("classification_report.pdf")