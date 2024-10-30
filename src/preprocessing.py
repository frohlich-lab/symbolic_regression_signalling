import argparse
import pandas as pd

def merge_and_preprocess_datasets(train_path, test_path, valid_path, output_path, target_feature):
    """Merge train, test, and validation datasets into a single file, preprocess the data, and move the target feature to the last column."""
    # Load the datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    valid_data = pd.read_csv(valid_path)

    # Merge the datasets
    merged_data = pd.concat([train_data, test_data, valid_data])

    # Rename columns based on the given mapping
    merged_data = merged_data.rename({
        'K(p=None)': 'K',
        "P(phospho='u', k=None)": 'P_u',
        "P(phospho='p', k=None)": 'P_p',
        "K(p=1) % P(phospho='u', k=1)": "KPu",
        "koff_substrate": "k_off",
        "kD_substrate": "k_D",
        "kcat": "k_cat",
        "kinact": "k_inact",
        'K(p=None)': 'K',
        "P(phospho='u', k=None)":'P_u',
        "P(phospho='p', k=None)":'P_phospho_p_k_none',
        "K(p=1) % P(phospho='u', k=1)":'KPu',
        "dP()":'dP',
        "dK()":'dK',
    }, axis=1)

    # Create a new column 'tK' as the sum of 'K' and 'KPu'
    if 'K' in merged_data.columns and 'KPu' in merged_data.columns:
        merged_data['tK'] = merged_data['K'] + merged_data['KPu']

    # Ensure the target feature is the last column in the dataset
    if target_feature in merged_data.columns:
        # Move the target feature to the last position
        columns = [col for col in merged_data.columns if col != target_feature] + [target_feature]
        merged_data = merged_data[columns]

    # Save the preprocessed merged dataset to the output file
    merged_data.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Merge and preprocess train, test, and validation datasets')
    parser.add_argument('--train', required=True, help='Path to the train dataset CSV file')
    parser.add_argument('--test', required=True, help='Path to the test dataset CSV file')
    parser.add_argument('--valid', required=True, help='Path to the validation dataset CSV file')
    parser.add_argument('--output', required=True, help='Path to the output merged dataset CSV file')
    parser.add_argument('--target_feature', required=True, help='Name of the target feature to move to the last column')

    args = parser.parse_args()
    merge_and_preprocess_datasets(args.train, args.test, args.valid, args.output, args.target_feature)

if __name__ == '__main__':
    main()