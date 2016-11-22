from collections import defaultdict
from itertools import groupby, chain
from operator import itemgetter as get, attrgetter as by

from classes import CAR, DatasetObj


def prune(R_pos, R_neg, dataset):
    R_newpos, R_newneg = [], []
    R_pos = sorted(R_pos, key=by('confidence'), reverse=True)
    R_neg = sorted(R_neg, key=by('confidence'), reverse=True)

    # 1. Negative CARs pruning
    R_newneg = list(
        filter(lambda r:
               not any(r.antecedent <= t.antecedent and r.class_ == t.class_
                       for t in dataset),
               R_neg)
    )

    # 2. Positive CARs pruning
    for r in R_pos:
        if any(r.antecedent <= t.antecedent and r.class_ == t.class_ for t in dataset):
            R_newpos.append(r)
            dataset = filter(lambda t: not (r.antecedent <= t.antecedent and r.class_ == t.class_),
                             dataset)

    # 3. Negative CARs set adjustment
    if len(R_newneg) > len(R_newpos):
        R_newneg = R_newneg[:len(R_newpos)]

    return R_newpos, R_newneg


def predict(R_newpos, R_newneg, o, classes, classification_method='SUM') -> str:
    T_pos = filter(lambda r: r.antecedent <= o.antecedent, R_newpos)
    T_neg = map(lambda r: r._replace(confidence=-r.confidence),
                filter(lambda r: r.antecedent <= o.antecedent, R_newneg))

    Ts = groupby(sorted(chain(T_pos, T_neg), key=by('class_')), by('class_'))

    if classification_method == 'BEST':
        class_to_val_tuples = map(lambda k_v:
                                  (k_v[0], max(k_v[1], key=by('confidence')).confidence),
                                  Ts)
    elif classification_method == 'AVE':
        class_to_val_tuples = map(lambda k_v:
                                  (k_v[0], sum(map(by('confidence'), k_v[1])) / (len(k_v[1])) or 1),
                                  Ts)
    elif classification_method == 'SUM':
        class_to_val_tuples = map(lambda k_v:
                                  (k_v[0], sum(map(by('confidence'), k_v[1]))),
                                  Ts)
    else:
        raise ValueError("Invalid classification_method value (should be on of: 'BEST', 'SUM', 'AVE')")

    class_to_val = defaultdict(lambda: 0)
    class_to_val.update(dict(class_to_val_tuples))

    return max([(c, class_to_val[c]) for c in classes], key=get(1))[0]


if __name__ == '__main__':
    print(predict(R_newpos=[CAR({'23'}, '90', 1)], R_newneg=[CAR({'22'}, '90', .972)],
                  o=DatasetObj(set('1 10 18 20 22 25 26 28 31 39 42 47 51 55 60 61 63 66 71 73 77 84'.split())),
                  classes={'89', '90'}))
