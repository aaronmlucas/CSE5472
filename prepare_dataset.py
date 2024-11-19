import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing feature labels
with open('./data/kddcup.names', 'r') as f:
    f.readline() # Skip first line. We grab the attack classifications from a different file.
    # Grab all feature labels. Split along ':' to remove type information.
    feature_labels = [line.split(':')[0] for line in f.readlines()]
    feature_labels.append('target') # Add target as the last feature since KDDCup dataset has that but the names file does not.
    
# Importing attack classifications
with open('./data/training_attack_types', 'r') as f:
    # Create a dictionary of attack types. 
    attack_types = dict([line.strip().split(' ') for line in f.readlines() if line.strip()])
    attack_types['normal'] = 'normal' # Add standard/safe traffic to the dictionary.
    
dataset = pd.read_csv('./data/kddcup.data_10_percent.gz', names=feature_labels)
# Use the attack_types dictionary to add a new column to the dataset that maps the target column (key) to the attack type (value).
dataset['attack_type'] = dataset['target'].apply(lambda x: attack_types[x[:-1]])

# We now have dataset, feature labels, and attack types. These are important for the rest of the project.