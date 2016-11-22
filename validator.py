import os

from utils import read_cars

if __name__ == '__main__':
    for dataset_name in sorted(os.listdir('rules')):
        print("Checking for '%s' ..." % dataset_name)
        for train_split in sorted(
                os.listdir(os.path.join('rules', dataset_name)),
                key=lambda f: int(f.replace('train', '').replace('_rules.txt', ''))):
            pos_rules, neg_rules = read_cars(os.path.join('rules', dataset_name, train_split))
            print('    %s:' % train_split)
            if not pos_rules:
                print('        No POSITIVE rules')
            if not neg_rules:
                print('        No NEGATIVE rules')
            print('        # POS: %d    #NEG: %d' % (len(pos_rules), len(neg_rules)))
