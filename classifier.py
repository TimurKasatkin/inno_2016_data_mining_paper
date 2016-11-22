from collections import defaultdict
from itertools import groupby, chain
from operator import itemgetter as get, attrgetter as by


class ProposedClassifier:
    def __init__(self, R_pos, R_neg, dataset, classes=None):
        self.classes = classes if classes else set(map(by('class_'), dataset))
        self.R_newpos, self.R_newneg = self.__prune(R_pos, R_neg, dataset)

    def predict(self, o, classification_method='SUM', only_positive=False):
        T_pos = filter(lambda r: r.antecedent <= o.antecedent, self.R_newpos)

        if classification_method == 'BEST':
            return max(T_pos, key=by('confidence')).class_
        else:
            if only_positive:
                T_neg = []
            else:
                T_neg = map(lambda r: r._replace(confidence=-r.confidence),
                            filter(lambda r: r.antecedent <= o.antecedent, self.R_newneg))

            Ts = groupby(sorted(chain(T_pos, T_neg), key=by('class_')), by('class_'))
            if classification_method == 'AVE':
                class_to_score_tuples = \
                    map(lambda k_v:
                        (lambda v: (k_v[0], sum(map(by('confidence'), v)) / (len(v) or 1)))(list(k_v[1])),
                        Ts)
            elif classification_method == 'SUM':
                class_to_score_tuples = map(lambda k_v:
                                            (k_v[0], sum(map(by('confidence'), k_v[1]))),
                                            Ts)
            else:
                raise ValueError("Invalid classification_method value (should be on of: 'BEST', 'SUM', 'AVE')")

        class_score = defaultdict(lambda: 0)
        class_score.update(dict(class_to_score_tuples))

        return max([(c, class_score[c]) for c in self.classes], key=get(1))[0]

    @staticmethod
    def __prune(R_pos, R_neg, dataset):
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
        R_newpos = []
        for rule in R_pos:
            if any(rule.antecedent <= t.antecedent and rule.class_ == t.class_ for t in dataset):
                R_newpos.append(rule)
                dataset = list(filter(lambda t: not (rule.antecedent <= t.antecedent and rule.class_ == t.class_),
                                      dataset))

        # 3. Negative CARs set adjustment
        if len(R_newneg) > len(R_newpos):
            R_newneg = R_newneg[:len(R_newpos)]

        return R_newpos, R_newneg
