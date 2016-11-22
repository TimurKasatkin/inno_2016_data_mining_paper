import re

from classes import DatasetObj, CAR


def read_dataset(filepath):
    with open(filepath, 'r') as f:
        return list(
            map(lambda split: DatasetObj(set(split[:-1]), split[-1]),
                map(lambda l: list(map(int, l.split())), f))
        )


__rule_regexp = re.compile('(?P<antecedent>(\S+ )+)-> (?P<class>~?\S+)')
__conf_regexp = re.compile('cf=(?P<confidence>\d\.?\d*)')


def read_cars(filename):
    positive_rules, negative_rules = [], []

    with open(filename, 'r') as f:
        for line in f:
            m = __rule_regexp.search(line)
            antecedent = set(map(int, m.group('antecedent').split()))
            class_ = m.group('class')
            confidence = float(__conf_regexp.search(line).group('confidence'))
            if class_.startswith('~'):
                class_ = class_[1:]
                cur_list = negative_rules
            else:
                cur_list = positive_rules
            cur_list.append(CAR(antecedent, int(class_), confidence))

        return positive_rules, negative_rules


if __name__ == '__main__':
    cars1_positive, cars1_negative = read_cars('rules/adult/train1_rules.txt')
    print("Positive:")
    print(*cars1_positive, sep='\n')
    print("Negative:")
    print(*cars1_negative, sep='\n')
    print(len(cars1_positive), len(cars1_negative))
