import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import sys
from pathlib import Path

def evaluate_predictions(actual_data_path: str, predictions_path: str):
    """
    Compares predictions with actual values and prints evaluation metrics.

    Args:
        actual_data_path: Path to the CSV file with original data including the 'default' column.
        predictions_path: Path to the CSV file with model predictions.
    """
    print("=" * 60)
    print("Prediction Evaluation")
    print("=" * 60)

    try:
        # Load actual data
        actual_df = pd.read_csv(actual_data_path)
        if 'customer_ref' in actual_df.columns:
            actual_df = actual_df.rename(columns={'customer_ref': 'customer_id'})
        
        if 'default' not in actual_df.columns:
            print(f"Error: 'default' column not found in actual data file: {actual_data_path}")
            sys.exit(1)
            
        actuals = actual_df[['customer_id', 'default']]

        # Load predictions data (try both comma and tab separators)
        try:
            predictions_df = pd.read_csv(predictions_path, sep='\t')
        except:
            predictions_df = pd.read_csv(predictions_path)
            
        if 'default' not in predictions_df.columns:
            print(f"Error: 'default' column not found in predictions file: {predictions_path}")
            sys.exit(1)

        # Merge actual and predicted values
        merged_df = pd.merge(actuals, predictions_df, on='customer_id', suffixes=('_actual', '_predicted'))

        if merged_df.empty:
            print("Error: No matching customer_ids found between actual and predicted data.")
            sys.exit(1)

        y_actual = merged_df['default_actual']
        y_predicted = merged_df['default_predicted']

        # Calculate and print metrics
        accuracy = accuracy_score(y_actual, y_predicted)
        report = classification_report(y_actual, y_predicted, zero_division=0)
        cm = confusion_matrix(y_actual, y_predicted)

        print(f"Evaluation based on {len(merged_df)} records.")
        print(f"\nAccuracy: {accuracy:.4f}\n")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        print("\n(Rows: Actual, Columns: Predicted)")

        # Find and save mismatches
        mismatches = merged_df[merged_df['default_actual'] != merged_df['default_predicted']]
        if not mismatches.empty:
            mismatch_file = 'outputs/prediction_mismatches.csv'
            mismatches[['customer_id', 'default_actual', 'default_predicted']].to_csv(mismatch_file, index=False)
            print(f"\nFound {len(mismatches)} prediction mismatches.")
            print(f"Mismatches saved to {mismatch_file}")
        else:
            print("\nNo prediction mismatches found. Excellent!")
        
        print("=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Evaluate credit scoring predictions.')
    parser.add_argument(
        '--actual',
        required=True,
        help='Path to the CSV file with actual default values (e.g., data/application_metadata.csv)'
    )
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path to the CSV file with model predictions (e.g., predictions.csv)'
    )
    args = parser.parse_args()

    evaluate_predictions(args.actual, args.predictions)

if __name__ == '__main__':
    main()
