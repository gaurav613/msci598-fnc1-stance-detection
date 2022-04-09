## MSCI 598 - Final Project ##
## Gaurav Mudbhatkal - 20747018 ##

import csv


def get_weights(training_set):
    """
    method to obtain a weight for each stance in the corresponding training set(will be different for each epoch)
    Requires:
        train_dataset: FNCDataset object for training set
    Returns:
        a weight for each stance 
    """

    # initialize the count for each stance
    total_stances = 0
    stance_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    # update the counts

    for headline_body_pair, stance in training_set:
        stance_counts[stance] = stance_counts[stance] + 1
        total_stances += 1

    # dealing with issue of 0 weights
    if stance_counts[0] == 0:
        stance_counts[0] += 1
        total_stances += 1
    if stance_counts[1] == 0:
        stance_counts[1] += 1
        total_stances += 1
    if stance_counts[2] == 0:
        stance_counts[2] += 1
        total_stances += 1
    if stance_counts[3] == 0:
        stance_counts[3] += 1
        total_stances += 1

    return [total_stances / stance_counts[stance] for text, stance in training_set]


def save_predictions(pred, file):

    """
    Save predictions to CSV file
    Args:
        pred: list, of dictionaries
        file: str, filename + extension
    """

    with open(file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Headline', 'Body ID', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for prediction in pred:
            writer.writerow(prediction)