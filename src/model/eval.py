import torch
import evaluate
import time
import random
import numpy as np



def compute_lpa(prediction, ground_truth):
    """
    Computes the Location Prediction Accuracy (LPA) between a predicted and ground truth segmented string.

    Parameters:
    -----------
    prediction : str
        A predicted segmented string with words separated by spaces (e.g., "राम चन्द्रः").
    ground_truth : str
        A ground truth segmented string with words separated by spaces (e.g., "राम चन्द्रः").

    Returns:
    --------
    float
        The Location Prediction Accuracy (LPA).
    """
    def get_boundaries(segmented_word):
        # Get indices of spaces as boundaries
        return [i for i, char in enumerate(segmented_word) if char == ' ']

    pred_boundaries = get_boundaries(prediction)
    gt_boundaries = get_boundaries(ground_truth)

    total_correct = len(set(pred_boundaries) & set(gt_boundaries))  # Correctly predicted boundaries
    total_boundaries = len(gt_boundaries)  # Total ground truth boundaries

    lpa = total_correct / total_boundaries if total_boundaries > 0 else 0
    return lpa


def compute_spa(prediction, ground_truth):
    """
    Computes the Split Prediction Accuracy (SPA) between a predicted and ground truth segmented string.

    Parameters:
    -----------
    prediction : str
        A predicted segmented string with words separated by spaces (e.g., "राम चन्द्रः").
    ground_truth : str
        A ground truth segmented string with words separated by spaces (e.g., "राम चन्द्रः").

    Returns:
    --------
    float
        The Split Prediction Accuracy (SPA).
    """
    def get_segments(segmented_word):
        # Split the string by spaces to get segments
        return segmented_word.split()

    pred_segments = get_segments(prediction)
    gt_segments = get_segments(ground_truth)

    # Correctly predicted segments
    total_correct = len(set(pred_segments) & set(gt_segments))

    spa = total_correct/len(gt_segments) if len(gt_segments) > 0 else 0

    return spa



# def calculate_sws_metrics(df, prediction_col, target_col):
#     """
#     Calculates Precision, Recall, F1 Score, and Perfect Match (PM) 
#     for Sanskrit Word Segmentation (SWS) from a DataFrame.

#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         The DataFrame containing the predictions and targets.
#     prediction_col : str
#         The name of the column containing the predicted segmented strings.
#     target_col : str
#         The name of the column containing the ground truth segmented strings.

#     Returns:
#     --------
#     dict
#         A dictionary containing Precision, Recall, F1 Score, and Perfect Match (PM).
#     """

#     # Normalize spaces in the prediction and target columns
#     df[prediction_col] = df[prediction_col].apply(lambda x: ' '.join(x.split()))
#     df[target_col] = df[target_col].apply(lambda x: ' '.join(x.split()))

#     # Convert strings to sets of tokens
#     df['pred_tokens'] = df[prediction_col].apply(lambda x: set(x.split()))
#     df['target_tokens'] = df[target_col].apply(lambda x: set(x.split()))

#     # Calculate true positives, false positives, and false negatives for each row
#     df['true_positives'] = df.apply(lambda row: len(row['pred_tokens'] & row['target_tokens']), axis=1)
#     df['false_positives'] = df.apply(lambda row: len(row['pred_tokens'] - row['target_tokens']), axis=1)
#     df['false_negatives'] = df.apply(lambda row: len(row['target_tokens'] - row['pred_tokens']), axis=1)
#     print(df)

#     # Calculate aggregate true positives, false positives, and false negatives
#     total_true_positives = df['true_positives'].sum()
#     total_false_positives = df['false_positives'].sum()
#     total_false_negatives = df['false_negatives'].sum()

#     # Calculate Precision, Recall, F1 Score
#     precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
#     recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     # Calculate Perfect Match (PM)
#     perfect_matches = np.sum(df[prediction_col] == df[target_col])
#     pm = perfect_matches / len(df) if len(df) > 0 else 0

#     return {
#         'Precision': precision * 100,
#         'Recall': recall * 100,
#         'F1 Score': f1 * 100,
#         'Perfect Match (PM)': pm * 100
#     }


def compute_sws_metrics(df, pred_col="predicted_labels", target_col="target_labels"):
    """
    Computes precision, recall, and F1 score for token classification from the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the predicted and actual labels.
    pred_col : str, optional
        The name of the column containing the predicted labels. Default is "predicted_labels".
    target_col : str, optional
        The name of the column containing the actual ground truth labels. Default is "labels".

    Returns:
    --------
    dict
        A dictionary containing the precision, recall, and F1 score.
    """

    # Load precision, recall, and F1 from evaluate library
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Vectorized sanity check to ensure that the lengths of predicted and target labels match
    lengths_match = (df[pred_col].apply(len) == df[target_col].apply(len)).all()

    if not lengths_match:
        raise ValueError("Mismatch in lengths between predicted labels and actual labels.")

    print("Sanity check passed: All predictions and labels have matching lengths.")

    # Use numpy to flatten the lists of predicted labels and ground truth labels
    predictions_flat = np.concatenate(df[pred_col].values)
    labels_flat = np.concatenate(df[target_col].values)

    # Compute metrics using the evaluate library
    precision = precision_metric.compute(predictions=predictions_flat, references=labels_flat, average=None)["precision"]
    recall = recall_metric.compute(predictions=predictions_flat, references=labels_flat, average=None)["recall"]
    f1 = f1_metric.compute(predictions=predictions_flat, references=labels_flat, average=None)["f1"]


    # Calculate Perfect Match (PM)
    perfect_matches = np.sum(df[pred_col] == df[target_col])
    pm = perfect_matches / len(df) if len(df) > 0 else 0

    return {
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        'Perfect Match (PM)': pm * 100
    }




