import argparse
from collections import Counter
from sys import argv

import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    'mode',
    help='possible values:\n' +
         '\t - verify   (compare against ground truth)\n' +
         '\t - compare  (compare against other result file)\n' +
         '\t - evaluate (compute accuracy of single file)'
)
parser.add_argument('-f1')
parser.add_argument('-f2')
args = parser.parse_args()


def read_ground_truth(file_name):
    ground_truth = []

    with open(file_name) as f:
        for line in f:
            if 'not considered' in line:
                ground_truth.append(None)
            elif 'verified label' in line:
                ground_truth.append(1)
            elif 'failed label' in line:
                ground_truth.append(0)
            elif 'analysis precision' in line:
                continue
            else:
                raise RuntimeError('unknown case in ground truth file')

    return ground_truth


def read_results_file(file_name):
    analyzer_results = []
    runtimes = []

    with open(file_name) as f:
        idx = 0

        for line in f:
            if 'Academic license' in line:
                continue
            elif 'image not correctly classified' in line:
                analyzer_results.append(None)
            elif line == 'verified\n':
                analyzer_results.append(1)
            elif line == 'can not be verified\n':
                analyzer_results.append(0)
            elif 'analysis time' in line:
                runtime = float(line.split('time: ')[1].split(' seconds')[0])
                runtimes.append(runtime)
                idx += 1

                if 6 * 60 <= runtime:
                    print(f'Image {idx}: LP ran for {runtime / 60} minutes')
            else:
                raise RuntimeError('unknown case in analyzer results file')

    return analyzer_results, runtimes


def compare_results(ground_truth, analyzer_results):
    assert len(ground_truth) == len(analyzer_results)

    def misclassified(results):
        return [i for i, j in enumerate(results) if j is None]

    print()
    assert misclassified(ground_truth) == misclassified(analyzer_results)
    print(len(misclassified(ground_truth)), 'misclassifed images')

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    idx = 0
    for true_val, computed_val in zip(ground_truth, analyzer_results):
        if true_val is None:
            continue

        idx += 1
        if true_val == 1 and computed_val == 1:
            TP += 1
        elif true_val == 0 and computed_val == 1:
            FP += 1
        elif true_val == 1 and computed_val == 0:
            FN += 1
        elif true_val == 0 and computed_val == 0:
            TN += 1
        else:
            raise RuntimeError('unsupported truth value')

    print()
    print('True Positives: ', TP)
    print('False Positives:', FP)
    print('True Negatives: ', TN)
    print('False Negatives:', FN)
    print()

    assert FP == 0

    print(
        'Accuracy:',
        (TP + TN) / (len(ground_truth) - len(misclassified(ground_truth)))
    )
    print()


def print_runtimes(results, runtimes):
    success_runtimes = runtimes[results == 1]
    failure_runtimes = runtimes[results == 0]

    if success_runtimes.size:
        print(
            '- verified    :', round(np.mean(success_runtimes), 3),
            '+-', round(np.var(success_runtimes), 3),
            '( max =', round(np.max(success_runtimes), 3), ')'
        )
    if failure_runtimes.size:
        print(
            '- not verified:', round(np.mean(failure_runtimes), 3),
            '+-', round(np.var(failure_runtimes), 3),
            '( max =', round(np.max(failure_runtimes), 3), ')'
        )
    print()


if args.mode == 'verify':
    if len(argv) != 6:
        print(
            'python3 evaluate_analyzer.py verify' +
            '-f1 ground_truth.txt -f2 results_file.txt'
        )
        exit(1)

    ground_truth = read_ground_truth(args.f1)
    analyzer_results = read_results_file(args.f2)[0]
    compare_results(ground_truth, analyzer_results)

elif args.mode == 'compare':
    if len(argv) != 6:
        print(
            'python3 evaluate_analyzer.py compare' +
            '-f1 results_file_1.txt -f2 results_file_2.txt'
        )
        exit(1)

    ground_truth, old_runtimes = read_results_file(args.f1)
    analyzer_results, new_runtimes = read_results_file(args.f2)
    compare_results(ground_truth, analyzer_results)

    print('Runtimes F1:')
    print_runtimes(np.array(ground_truth), np.array(old_runtimes))
    print('Runtimes F2:')
    print_runtimes(np.array(analyzer_results), np.array(new_runtimes))


elif args.mode == 'evaluate':
    if len(argv) != 4:
        print(
            'python3 evaluate_analyzer.py evaluate -f1 results_file.txt'
        )
        exit(1)

    analyzer_results, runtimes = read_results_file(args.f1)
    eval_results = Counter(analyzer_results)
    print('verified:     ', eval_results[1])
    print('not verified: ', eval_results[0])
    print('misclassified:', eval_results[None])
    print()
    print_runtimes(np.array(analyzer_results), np.array(runtimes))

else:
    raise ValueError('unknown mode')
