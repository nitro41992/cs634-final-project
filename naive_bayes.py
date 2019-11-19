import csv


def get_labels(filename):
    with open(filename, "rt", encoding='utf8') as f:
        file = csv.reader(f)
        temp = list(file)
    return temp
