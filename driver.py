#!/usr/bin/env python

import numpy as np, os, sys
from get_sepsis_score import load_sepsis_model, get_sepsis_score
# Nawawy's start
import joblib
# Nawawy's end

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Nawawy's start
    if not os.path.isdir(input_directory):
        raise FileNotFoundError(f"Directory does not exist: {input_directory}")

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'Predictions', 'Benign'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'Predictions', 'Adversarial'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'Data', 'Benign'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'Data', 'Adversarial'), exist_ok=True)
    # Nawawy's end

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    model = load_sepsis_model()

    # Iterate over files.
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)
        # Make predictions.
        num_rows = len(data)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        # Nawawy's start
        scores_adversarial = np.zeros(num_rows)
        labels_adversarial = np.zeros(num_rows)
        adversarial_data = np.random.rand(data.shape[0], data.shape[1])
        # Nawawy's end
        for t in range(num_rows):
            current_data = data[:t+1]
            current_score, current_label = get_sepsis_score(current_data, model)
            scores[t] = current_score
            labels[t] = current_label
            # Nawawy's start
            current_score_adversarial, current_label_adversarial = get_sepsis_score(current_data, model, adversary=True, adversarial_data=adversarial_data)
            scores_adversarial[t] = current_score_adversarial
            labels_adversarial[t] = current_label_adversarial
            # Nawawy's end

        # Save results.
        # Nawawy's start
        output_file = os.path.join(output_directory, 'Predictions', 'Benign', f)
        # Nawawy's end
        save_challenge_predictions(output_file, scores, labels)

        # Nawawy's start
        output_file = os.path.join(output_directory, 'Predictions', 'Adversarial', f)
        save_challenge_predictions(output_file, scores_adversarial, labels_adversarial)
        joblib.dump(data, os.path.join(output_directory, 'Data', 'Benign')+'/'+f[:-4]+'.pkl')
        joblib.dump(adversarial_data, os.path.join(output_directory, 'Data', 'Adversarial') + '/' + f[:-4] + '.pkl')
        # Nawawy's end
