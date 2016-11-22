from collections import namedtuple

DatasetObj = namedtuple("Object", ["antecedent", "class_"])
DatasetObj.__new__.__defaults__ = (None,) * len(DatasetObj._fields)

CAR = namedtuple("CAR", ["antecedent", "class_", "confidence"])
