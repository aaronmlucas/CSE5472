import pandas as pd
import gzip
import argparse

def fetch_feature_names(local_file):
    """
    Fetch and parse feature names from the local kddcup.names file.
    
    Args:
        local_file (str): Path to the kddcup.names file.
    
    Returns:
        list: List of feature names.
    """
    try:
        with open(local_file, 'r') as file:
            lines = file.readlines()
        feature_names = []
        for line in lines:
            if not line.startswith('|') and ':' in line:
                feature_name = line.split(':')[0].strip()
                feature_names.append(feature_name)
        # Add the 'label' column as the last feature
        feature_names.append('label')
        return feature_names
    except Exception as e:
        print(f"An error occurred while fetching feature names: {e}")
        return []

def extract_unique_values(input_gz, output_file, feature_names_file):
    """
    Extract unique values for each feature in the KDDCup dataset from a compressed .gz file.
    
    Args:
        input_gz (str): Path to the KDDCup dataset .gz file.
        output_file (str): Path to save the extracted unique values as a text file.
        feature_names_file (str): Path to the kddcup.names file.
    """
    try:
        # Fetch feature names
        print(f"Fetching feature names from {feature_names_file}...")
        feature_names = fetch_feature_names(feature_names_file)
        if not feature_names:
            print("Failed to fetch feature names. Exiting.")
            return

        # Load the compressed dataset
        print(f"Loading dataset from {input_gz}...")
        with gzip.open(input_gz, 'rt') as gz_file:
            data = pd.read_csv(gz_file, header=None, names=feature_names)

        # Extract unique values for each feature
        print("Extracting unique values for each feature...")
        unique_values = {col: data[col].unique().tolist() for col in data.columns}

        # Save the results to a file
        with open(output_file, 'w') as file:
            for feature, values in unique_values.items():
                file.write(f"Feature: {feature}\n")
                file.write(f"Unique Values ({len(values)}):\n{values}\n\n")
        
        print(f"Unique values successfully extracted and saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unique values for each feature in the KDDCup dataset from a .gz file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the KDDCup dataset .gz file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the unique values as a text file.")
    parser.add_argument("-f", "--features", required=True, help="Path to the local kddcup.names file.")
    args = parser.parse_args()

    extract_unique_values(args.input, args.output, args.features)
