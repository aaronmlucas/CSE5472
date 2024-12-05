import time
import os
import joblib
import numpy as np
import pandas as pd
import prepare_dataset
from sklearn.utils import shuffle
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
# Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

debugging = False  # Enable sanity checks if True

def train_gaussian_nb(training_X, training_Y, testing_X, testing_Y, model_metrics):
    nb = GaussianNB()
    start_time = time.time()
    nb.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(nb, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Gaussian Naive Bayes"] = metrics
    return nb


def train_decision_tree(training_X, training_Y, testing_X, testing_Y, model_metrics):
    tree = DecisionTreeClassifier()
    start_time = time.time()
    tree.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(tree, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Decision Tree"] = metrics
    return tree


def train_random_forest(training_X, training_Y, testing_X, testing_Y, model_metrics):
    forest = RandomForestClassifier()
    start_time = time.time()
    forest.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(forest, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Random Forest"] = metrics
    return forest


def train_gradient_boosting(training_X, training_Y, testing_X, testing_Y, model_metrics):
    gb = GradientBoostingClassifier()
    start_time = time.time()
    gb.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(gb, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Gradient Boosting"] = metrics
    return gb


def train_svm(training_X, training_Y, testing_X, testing_Y, model_metrics):
    svm = SVC(probability=True)
    start_time = time.time()
    svm.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(svm, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["SVM"] = metrics
    return svm


def train_logistic_regression(training_X, training_Y, testing_X, testing_Y, model_metrics):
    lr = LogisticRegression()
    start_time = time.time()
    lr.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(lr, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Logistic Regression"] = metrics
    return lr


def train_neural_network(training_X, training_Y, testing_X, testing_Y, model_metrics):
    # Ensure Y is one-hot encoded
    if training_Y.ndim == 1:  # Encode only if not already one-hot
        training_Y = to_categorical(training_Y)
    if testing_Y.ndim == 1:
        testing_Y = to_categorical(testing_Y)

    def build_neural_network():
        num_classes = training_Y.shape[1]  # Dynamically set output layer
        nn = Sequential()
        nn.add(Dense(training_X.shape[1], input_dim=training_X.shape[1], activation='relu'))
        nn.add(Dense(num_classes, activation='softmax'))
        nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return nn

    nn = build_neural_network()
    start_time = time.time()
    nn.fit(training_X, training_Y, epochs=10, batch_size=10, verbose=1)
    end_time = time.time()

    metrics = collect_nn_metrics(nn, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Neural Network"] = metrics
    return nn

def split_data(dataset):
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    if Y.dtypes == 'object':
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
    
    X = MinMaxScaler().fit_transform(X)
    return train_test_split(X, Y, test_size=0.2, random_state=777)


def collect_model_metrics(model, training_X, training_Y, testing_X, testing_Y):
    metrics = {}
    training_preds = model.predict(training_X)
    testing_preds = model.predict(testing_X)
    
    metrics['Training Accuracy'] = accuracy_score(training_Y, training_preds)
    metrics['Training Precision'] = precision_score(training_Y, training_preds, average='weighted', zero_division=0)
    metrics['Training Recall'] = recall_score(training_Y, training_preds, average='weighted', zero_division=0)
    metrics['Training F1'] = f1_score(training_Y, training_preds, average='weighted')
    
    metrics['Testing Accuracy'] = accuracy_score(testing_Y, testing_preds)
    metrics['Testing Precision'] = precision_score(testing_Y, testing_preds, average='weighted', zero_division=0)
    metrics['Testing Recall'] = recall_score(testing_Y, testing_preds, average='weighted', zero_division=0)
    metrics['Testing F1'] = f1_score(testing_Y, testing_preds, average='weighted')
    
    metrics['Confusion Matrix'] = confusion_matrix(testing_Y, testing_preds).tolist()
    print(f"Model Metrics: {metrics}")
    return metrics

def collect_nn_metrics(model, training_X, training_Y, testing_X, testing_Y):
    metrics = {}
    training_preds = model.predict(training_X).argmax(axis=1)  # Convert probabilities to class indices
    testing_preds = model.predict(testing_X).argmax(axis=1)
    
    # Handle one-hot encoded Y values
    if training_Y.ndim > 1:
        training_Y = training_Y.argmax(axis=1)
    if testing_Y.ndim > 1:
        testing_Y = testing_Y.argmax(axis=1)
    
    metrics['Training Accuracy'] = accuracy_score(training_Y, training_preds)
    metrics['Training Precision'] = precision_score(training_Y, training_preds, average='weighted', zero_division=0)
    metrics['Training Recall'] = recall_score(training_Y, training_preds, average='weighted', zero_division=0)
    metrics['Training F1'] = f1_score(training_Y, training_preds, average='weighted')
    
    metrics['Testing Accuracy'] = accuracy_score(testing_Y, testing_preds)
    metrics['Testing Precision'] = precision_score(testing_Y, testing_preds, average='weighted', zero_division=0)
    metrics['Testing Recall'] = recall_score(testing_Y, testing_preds, average='weighted', zero_division=0)
    metrics['Testing F1'] = f1_score(testing_Y, testing_preds, average='weighted')
    
    metrics['Confusion Matrix'] = confusion_matrix(testing_Y, testing_preds).tolist()
    print(f"Neural Network Metrics: {metrics}")
    return metrics


def sanity_check_all_models(models, training_X, testing_X, training_Y, testing_Y):
    print("\nSanity Check with Shuffled Labels:")
    shuffled_training_Y = shuffle(training_Y)
    shuffled_testing_Y = shuffle(testing_Y)

    results = {}
    for model_name, model_function in models.items():
        print(f"Running sanity check for {model_name}...")
        model = model_function()  # Create the model instance
        model.fit(training_X, shuffled_training_Y)  # Train on shuffled labels
        shuffled_accuracy = model.score(testing_X, shuffled_testing_Y)  # Evaluate
        results[model_name] = shuffled_accuracy
        print(f"Sanity Check Accuracy for {model_name}: {shuffled_accuracy}")

    return results


def get_feature_importances(trained_models, feature_names):
    # Note: This only works for tree-based models
    feature_importances = {}
    for model_name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):  # For tree-based models
            importances = model.feature_importances_
            feature_importances[model_name] = dict(zip(feature_names, importances))
    return feature_importances


def get_dataset_stats(dataset):
    class_counts = dataset.iloc[:, -1].value_counts()
    stats = {
        "Total Samples": dataset.shape[0],
        "Total Features": dataset.shape[1] - 1,
        "Class Distribution": class_counts.to_dict()
    }
    return stats

def generate_pdf_report(metrics, trained_models, feature_importances, dataset_stats, append_to_pdf=False):
    report_file = "model_metrics_report.pdf"
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Add general dataset statistics
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt="Dataset Statistics:", ln=True)
    pdf.set_font("Arial", size=12)
    for stat, value in dataset_stats.items():
        pdf.cell(0, 10, txt=f"{stat}: {value}", ln=True)
    pdf.ln(10)

    # Add metrics for each model
    for model_name, model_metrics in metrics.items():
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, txt=f"Metrics for {model_name}:", ln=True)
        pdf.set_font("Arial", size=12)
        for metric, value in model_metrics.items():
            if metric == "Confusion Matrix":
                continue
            
            pdf.cell(0, 10, txt=f"{metric}: {value}", ln=True)

        # Add confusion matrix plot
        if "Confusion Matrix" in model_metrics:
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                model_metrics["Confusion Matrix"],
                annot=True,
                fmt="d",
                cmap="Blues"
            )
            plt.title(f"Confusion Matrix: {model_name}")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plot_path = f"{model_name.replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            pdf.image(plot_path, x=10, y=None, w=180)
            pdf.ln(10)
            os.remove(plot_path)

        pdf.ln(5)

    # Add feature importances
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, txt="Feature Importances (Tree-Based Models Only):", ln=True)
    for model_name, importances in feature_importances.items():
        if importances is None:
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt=f"{model_name}: Feature importances not available.", ln=True)
        else:
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, txt=f"Model: {model_name}", ln=True)
            pdf.set_font("Arial", size=10)
            for feature, importance in sorted(importances.items(), key=lambda x: -x[1]):
                pdf.cell(0, 10, txt=f"  {feature}: {importance:.4f}", ln=True)
            pdf.ln(5)

    # Save the PDF
    pdf.output(report_file)
    print(f"Metrics report saved to '{report_file}'.")


def select_and_train_models(model_list, model_metrics, training_X, training_Y, testing_X, testing_Y):
    available_models = {
        "Gaussian Naive Bayes": lambda: train_gaussian_nb(training_X, training_Y, testing_X, testing_Y, model_metrics),
        "Decision Tree": lambda: train_decision_tree(training_X, training_Y, testing_X, testing_Y, model_metrics),
        "Random Forest": lambda: train_random_forest(training_X, training_Y, testing_X, testing_Y, model_metrics),
        "Gradient Boosting": lambda: train_gradient_boosting(training_X, training_Y, testing_X, testing_Y, model_metrics),
        "SVM": lambda: train_svm(training_X, training_Y, testing_X, testing_Y, model_metrics),
        "Logistic Regression": lambda: train_logistic_regression(training_X, training_Y, testing_X, testing_Y, model_metrics),
        "Neural Network": lambda: train_neural_network(training_X, training_Y, testing_X, testing_Y, model_metrics),
    }

    if model_list is None:
        print("Available models:")
        for idx, model_name in enumerate(available_models.keys(), start=1):
            print(f"{idx}. {model_name}")
        
        selected = input("Enter the numbers of the models to train (comma-separated): ")
        selected_indices = [int(i.strip()) for i in selected.split(",")]
        model_list = [list(available_models.keys())[i - 1] for i in selected_indices]
    
    trained_models = {}
    for model_name in model_list:
        if model_name in available_models:
            print(f"Training {model_name}...")
            model = available_models[model_name]()
            trained_models[model_name] = model
        else:
            print(f"Model '{model_name}' not recognized.")
    
    print("Training completed.")
    return trained_models

if __name__ == "__main__":
    # Check if models folder exists and has saved models
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)

    print("Welcome to the KDDCup 1999 Model Trainer and Reporter!")
    print("Choose an action:")
    print("1. Train models on the dataset")
    print("2. Generate a report on existing models in the 'models' folder")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # Prepare dataset
        dataset = prepare_dataset.dataset
        training_X, testing_X, training_Y, testing_Y = split_data(dataset)
        
        feature_names = dataset.columns[:-1]  # Feature names for importance extraction
        model_metrics = {}  # Dictionary to store metrics for all models
        models_to_train = None  # Optionally specify models, or use interactive selection

        # Train selected models
        trained_models = select_and_train_models(models_to_train, model_metrics, training_X, training_Y, testing_X, testing_Y)

        # Generate dataset stats and feature importances
        dataset_stats = get_dataset_stats(dataset)
        feature_importances = get_feature_importances(trained_models, feature_names)

        # Save trained models
        for model_name, model in trained_models.items():
            model_path = os.path.join(models_folder, f"{model_name.replace(' ', '_')}.joblib")
            joblib.dump(model, model_path)
            print(f"Model '{model_name}' saved to '{model_path}'.")

        print("Training completed.")
        generate_pdf_report(model_metrics, trained_models, feature_importances, dataset_stats)

    elif choice == "2":
        # Load existing models
        model_files = [f for f in os.listdir(models_folder) if f.endswith('.joblib')]
        if not model_files:
            print("No models found in the 'models' folder. Please train models first.")
        else:
            trained_models = {}
            for model_file in model_files:
                model_name = model_file.replace("_", " ").replace(".joblib", "")
                model_path = os.path.join(models_folder, model_file)
                trained_models[model_name] = joblib.load(model_path)
                print(f"Loaded model '{model_name}' from '{model_path}'.")

            # Generate dataset stats and feature importances
            dataset = prepare_dataset.dataset
            dataset_stats = get_dataset_stats(dataset)
            feature_names = dataset.columns[:-1]  # Feature names for importance extraction
            feature_importances = get_feature_importances(trained_models, feature_names)

            # Generate PDF report
            model_metrics = {name: {} for name in trained_models.keys()}  # Placeholder if metrics are missing
            generate_pdf_report(model_metrics, trained_models, feature_importances, dataset_stats)
            print("Report generation completed. Please note that metrics are not available for existing models.")
    
    else:
        print("Invalid choice. Please restart the program and enter 1 or 2.")
