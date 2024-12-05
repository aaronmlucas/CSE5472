Sebastian Bell & Aaron Lucas
### What we Built:
  The team has created a Network Intrusion Detection System (NIDS) using machine learning to detect potential intrusions. Our NIDS was created using Python, and several modules relating to machine learning. The program allows great customizability to users, allowing them to decide what model to use for machine learning, such as Gaussian Naive Bayes, Logistic Regression, and even a Neural Network. This was done through Scikit-Learn and TensorFlow/Keras. The user may also view the statistics of training for each model, and compare the performance of each model to choose the best model for their use case.

  For the data used for training, we decided on using KDD Cup 1999 Data, as it had a high amount of quality data, and is well-known and often used in training. Our project includes both the full KDDCup dataset and the 10% dataset. The data features numerous features for a given connection such as protocol_type, dst_host_count, and num_shells.

  When running _train_models.py_, the user is asked if they would like to train new models or create a report based off of existing models. Training the models will take some time, so leave the program in the background. Once the models have been trained, the program will automatically generate a PDF report named model_metrics_report.pdf. Training uses the KDD CUP 1999 Data as mentioned previously, which contains both “bad” connections (attacks, intrusions) and “good” connections (regular traffic). Within the generated PDF file, there are three main sections: dataset statistics, model metrics, and feature significance (for tree-based models). Note: if the user creates a report based off of existing models, the model metrics system will always be empty. This data is collected during training. To view the metrics for each model, you have to train new models.

### How to Run Our Project:
To begin, install the following dependencies using the following command:

`pip install -r requirements.txt`

Once you have installed all necessary dependencies run the following command:

`python3 train_models.py`

Follow the instructions within the program.

### Methodology of Evaluation:
  As mentioned previously a PDF file will be generated once _train_models.py_ has been run if the user has chosen to do so. Within the PDF file multiple metrics are present for viewing for a given model:
- Training Accuracies
- Training Precisions
- Training Recalls
- Training F1’s
- Testing Accuracies
- Testing Precisions 
- Testing Recalls
- Testing F1’s
- Training Times
- Confusion Matrices
- Feature Importances (for tree-based models)
  
### Results:
View _model_metrics_report.pdf_ included within this zip file.
This was the most recent result of training each model on the full KDDCup dataset.

### Discussion and Analysis of Results:
  The results of the most recent training shows near perfect training results with each model. While this may be considered a good thing, it may also be suspicious. It could also indicate that there are issues with our training methods/ dataset such as overfitting, label leakages, or some other problem.

  We suspect that the issue is not with overfitting due to sanity checks that were done. Label leakages can also be ruled out since the “Feature Importance” section of the metrics report breaks down how the features contribute to the classification. A clue of what might be going on also lies in the same section. Each model has a lot of excess features that don’t contribute to the classification. Getting rid of these may help these scores a bit. It is also evident that KDDCup’s class distribution is less than ideal. That may also have an effect on our results.

### Conclusion:
  Overall, this project was a great learning experience for both team members. There were many ups and downs. Many iterations and failed attempts. But we also learned about different machine learning models, to be careful when choosing datasets, learned about data manipulation techniques (MinMaxScaler, LabelEncoder), and possible challenges that may arise with each model.

  The team had numerous challenges relating to data collection and design. Originally, the team planned to include data collection in order to allow the user to view the model performance on their own collected data, however, we ran into some issues. While KDD Cup 1999 data was very feature rich, allowing us to train on quality data easily, we realized that many data collection programs like Wireshark don’t collect data to the granularity that our training data had, leaving us unsure of how to continue. We spent a few weeks attempting to pivot, switching to different data collection methods or programs, but we ultimately had to scrap the idea due to time constraints.

  Through each of these issues, we learned more and more about what it takes to create machine learning models. While this project may not be a shining example of a successful machine-learning-IDS, it was successful in teaching skills and knowledge that will be useful in our future careers.
