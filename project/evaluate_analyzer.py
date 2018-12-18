from sys import argv

if len(argv) != 3:
    print(f'usage: python3 {argv[0]} ground_truth.txt analyzer_results.txt')
    exit(1)

ground_truth = []

with open(argv[1]) as f:
    for line in f:
        if 'not considered' in line:
            ground_truth.append(None)
        elif 'verified label' in line:
            ground_truth.append(1)
        elif 'failed label' in line:
            ground_truth.append(0)
        else:
            RuntimeError('unknown case in ground truth file')

analyzer_results = []

with open(argv[2]) as f:
    for line in f:
        if 'Academic license' in line or 'analysis time' in line:
            continue
        elif 'image not correctly classified' in line:
            analyzer_results.append(None)
        elif line == 'verified\n':
            analyzer_results.append(1)
        elif line == 'can not be verified\n':
            analyzer_results.append(0)
        else:
            RuntimeError('unknown case in analyzer results file')

assert len(ground_truth) == len(analyzer_results)


def get_misclassified(results):
    return [i for i, j in enumerate(results) if j is None]


assert get_misclassified(ground_truth) == get_misclassified(analyzer_results)
print(len(get_misclassified(ground_truth)), 'images not considered')

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
        RuntimeError('unsupported truth value')

print()
print('True Positives: ', TP)
print('False Positives:', FP)
print('True Negatives: ', TN)
print('False Negatives:', FN)
print()

assert FP == 0

print(
    'Accuracy:',
    (TP + TN) / (len(ground_truth) - len(get_misclassified(ground_truth)))
)
