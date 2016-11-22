import os
from collections import OrderedDict
from operator import attrgetter as by
from os.path import join as join_path

import numpy as np
from sklearn.metrics import accuracy_score

from classifier import ProposedClassifier
from utils import read_cars, read_dataset

if __name__ == '__main__':
    splits_dirpath = 'splits'
    rules_dirpath = 'rules'

    datasets_splits_dict = dict(
        map(lambda dataset_name:
            (dataset_name, map(lambda i: (
                join_path(splits_dirpath, dataset_name, 'train%d.txt' % (i + 1)),
                join_path(splits_dirpath, dataset_name, 'test%d.txt' % (i + 1)),
            ), range(0, 10))),
            os.listdir(splits_dirpath))
    )

    mean_accuracy_dict = OrderedDict()
    for dataset_name in sorted(os.listdir(rules_dirpath)):
        print("Training classifiers for parts of dataset '%s'" % dataset_name)
        best_method_results = []
        both_rules_ave_results, both_rules_sum_results = [], []
        pos_rules_ave_results, pos_rules_sum_results = [], []
        for rules_file_name, (train_dataset_path, test_dataset_path) in zip(
                sorted(os.listdir(join_path(rules_dirpath, dataset_name)),
                       key=lambda path: int(path.replace('train', '').replace('_rules.txt', ''))),
                datasets_splits_dict[dataset_name]):
            print("    Training classifier for %s ... " % rules_file_name, end='')
            pos_rules, neg_rules = rules = read_cars(join_path(rules_dirpath, dataset_name, rules_file_name))
            classifier = ProposedClassifier(*rules, read_dataset(train_dataset_path))
            print("End. Classifying ... ", end='')
            test_dataset = read_dataset(test_dataset_path)
            y_true = np.fromiter(map(by('class_'), test_dataset), np.byte)
            best_method_results.append(
                accuracy_score(y_true, np.fromiter(map(classifier.predict, test_dataset), np.byte)))
            both_rules_sum_results.append(
                accuracy_score(y_true, np.fromiter(map(lambda o: classifier.predict(o, 'SUM'), test_dataset), np.byte)))
            both_rules_ave_results.append(
                accuracy_score(y_true, np.fromiter(map(lambda o: classifier.predict(o, 'AVE'), test_dataset), np.byte)))
            pos_rules_sum_results.append(
                accuracy_score(y_true,
                               np.fromiter(map(lambda o: classifier.predict(o, 'SUM', True), test_dataset), np.byte)))
            pos_rules_ave_results.append(
                accuracy_score(y_true,
                               np.fromiter(map(lambda o: classifier.predict(o, 'AVE', True), test_dataset), np.byte)))
            print('End.')
            print('        Accuracies:')
            print('        - BEST      : %f' % best_method_results[-1])
            print('        - POS SUM   : %f' % pos_rules_sum_results[-1])
            print('        - POS AVE   : %f' % pos_rules_ave_results[-1])
            print('        - BOTH SUM  : %f' % both_rules_sum_results[-1])
            print('        - BOTH AVE  : %f' % both_rules_ave_results[-1])
        print('  Mean accuracies:')
        mean_accuracy_dict[dataset_name] = {
            'BEST': np.mean(best_method_results),
            'POS SUM': np.mean(pos_rules_sum_results),
            'POS AVE': np.mean(pos_rules_ave_results),
            'BOTH SUM': np.mean(both_rules_sum_results),
            'BOTH AVE': np.mean(both_rules_ave_results)
        }
        print('    - BEST       : %f' % mean_accuracy_dict[dataset_name]['BEST'])
        print('    - POS SUM    : %f' % mean_accuracy_dict[dataset_name]['POS SUM'])
        print('    - POS AVE    : %f' % mean_accuracy_dict[dataset_name]['POS AVE'])
        print('    - BOTH SUM   : %f' % mean_accuracy_dict[dataset_name]['BOTH SUM'])
        print('    - BOTH AVE   : %f' % mean_accuracy_dict[dataset_name]['BOTH AVE'])
    with open('results.csv', 'w+') as res_f:
        res_f.write(',BEST,POS_SUM,POS_AVE,BOTH_SUM,BOTH_AVE%s' % os.linesep)
        for dataset_name, mean_accuracies in mean_accuracy_dict.items():
            res_f.write('%s, %.4f, %.4f4, %.4f, %.4f, %.4f%s' % (dataset_name,
                                                                 mean_accuracies['BEST'],
                                                                 mean_accuracies['POS SUM'],
                                                                 mean_accuracies['POS AVE'],
                                                                 mean_accuracies['BOTH SUM'],
                                                                 mean_accuracies['BOTH AVE'],
                                                                 os.linesep))
