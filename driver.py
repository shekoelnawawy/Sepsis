#!/usr/bin/env python

import numpy as np, os, sys
from get_sepsis_score import load_sepsis_model, get_sepsis_score

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Nawawy's start
    target_labels = np.nan
    # Nawawy's end
    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
    # Nawawy's start
        target_labels = data[:,-1]
        data = data[:, :-1]

    return data, target_labels
    # Nawawy's end

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
        # Nawawy's start
        data, target_labels = load_challenge_data(input_file)
        print('data')
        print(data)
        print(data.shape)
        print('--------------------------------------------')
        print('target_labels')
        print(target_labels)
        print(target_labels.shape)
        print('--------------------------------------------')
        i=0
        # Nawawy's end
        # Make predictions.
        num_rows = len(data)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        for t in range(num_rows):
            current_data = data[:t+1]
            # Nawawy's start
            print('current_data')
            print(current_data)
            print(current_data.shape)
            print('--------------------------------------------')
            # Nawawy's end
            current_score, current_label = get_sepsis_score(current_data, model)
            scores[t] = current_score
            labels[t] = current_label
            # Nawawy's start
            print('scores')
            print(scores)
            print(scores.shape)
            print('--------------------------------------------')
            print('labels')
            print(labels)
            print(labels.shape)
            print('--------------------------------------------')
            if i == 1:
                exit(1)
            i+=1
            # Nawawy's end

        # Save results.
        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)
