import time
import os
import joblib
import numpy as np
import pandas as pd
import prepare_dataset
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

debugging = True  # Change to true to receive additional debugging comments

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
    return metrics

def collect_nn_metrics(model, training_X, training_Y, testing_X, testing_Y):
    metrics = {}
    training_preds = model.predict(training_X).argmax(axis=1)
    testing_preds = model.predict(testing_X).argmax(axis=1)
    
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
    return metrics

def train_gaussian_nb(training_X, training_Y, testing_X, testing_Y, model_metrics):
    nb = GaussianNB()
    start_time = time.time()
    nb.fit(training_X, training_Y)
    end_time = time.time()
    
    metrics = collect_model_metrics(nb, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Gaussian Naive Bayes"] = metrics
    return nb

def build_neural_network(training_X, training_Y):
    nn = Sequential()
    nn.add(Dense(training_X.shape[1], input_dim=training_X.shape[1], activation='relu'))
    nn.add(Dense(len(np.unique(training_Y)), activation='softmax'))
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn

def train_neural_network(training_X, training_Y, testing_X, testing_Y, model_metrics):
    training_Y = to_categorical(training_Y)
    testing_Y = to_categorical(testing_Y)
    
    keras = build_neural_network(training_X, training_Y)
    start_time = time.time()
    keras.fit(training_X, training_Y, epochs=10, batch_size=10, verbose=1)
    end_time = time.time()
    
    metrics = collect_nn_metrics(keras, training_X, training_Y, testing_X, testing_Y)
    metrics['Training Time'] = end_time - start_time
    model_metrics["Neural Network"] = metrics
    return keras

def sanity_check_all_models(models, training_X, testing_X, training_Y, testing_Y):
    from sklearn.utils import shuffle

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

def select_and_train_models(model_list, model_metrics, training_X, training_Y, testing_X, testing_Y):
    available_models = {
        "Gaussian Naive Bayes": lambda: GaussianNB(),
        "Decision Tree": lambda: DecisionTreeClassifier(),
        "Random Forest": lambda: RandomForestClassifier(),
        "SVM": lambda: SVC(probability=True),
        "Logistic Regression": lambda: LogisticRegression(),
        "Gradient Boosting": lambda: GradientBoostingClassifier(),
        "Neural Network": lambda: build_neural_network(training_X, training_Y),
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
            model.fit(training_X, training_Y)
            metrics = collect_model_metrics(model, training_X, training_Y, testing_X, testing_Y)
            model_metrics[model_name] = metrics
            trained_models[model_name] = model
        else:
            print(f"Model '{model_name}' not recognized.")
    
    print("Training completed.")

    # Run sanity checks only if debugging is enabled
    if debugging:
        sanity_results = sanity_check_all_models(available_models, training_X, testing_X, training_Y, testing_Y)
        print("\nSanity Check Results:")
        for model_name, accuracy in sanity_results.items():
            print(f"Sanity Check Accuracy for {model_name}: {accuracy}")

    return trained_models

if __name__ == "__main__":
    dataset = prepare_dataset.dataset
    training_X, testing_X, training_Y, testing_Y = split_data(dataset)
    
    model_metrics = {}
    models_to_train = None
    trained_models = select_and_train_models(models_to_train, model_metrics, training_X, training_Y, testing_X, testing_Y)

    print_report = input("Would you like to print a report of the metrics for all trained models? (yes/no): ").strip().lower()
    if print_report == "yes":
        report_file = "model_metrics_report.txt"
        with open(report_file, "w") as f:
            f.write("Model Metrics Report:\n")
            for model_name, metrics in model_metrics.items():
                f.write(f"\nModel: {model_name}\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value}\n")
        print(f"Metrics report saved to '{report_file}'.")

    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)
    for model_name, model in trained_models.items():
        model_path = os.path.join(models_folder, f"{model_name.replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        print(f"Model '{model_name}' saved to '{model_path}'.")
