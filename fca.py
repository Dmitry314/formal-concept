import numpy as np
from copy import copy

class FCA:
    # object and attribute names need to be converted into numerals
    def __init__(self, data):
        self.data = data.copy().astype(dtype='bool')
        self.num_obj, self.num_attr = self.data.shape

        self.objects = np.array(range(self.num_obj))
        self.attributes = np.array(range(self.num_attr))

        self.lattice = {}

    def inverse_matrix(self):
        self.data = np.logical_not(self.data)

    def sort_columns(self):
        frequency = self.data.sum(axis=0)
        indices = np.argsort(frequency)[::-1]
        self.data = self.data[:, indices]

    def calculate_close(self, attrs):
        if len(attrs) == 0:
            return self.attributes[np.all(self.data, axis=0)]

        objs_ind = self.data[:, attrs[0]]
        for attr in attrs[1:]:
            objs_ind = np.logical_and(objs_ind, self.data[:, attr])
        objs = self.objects[np.asarray(objs_ind).squeeze()]

        if len(objs) == 0:
            attrs_cl = self.attributes.copy()
            return objs, attrs_cl

        attrs_ind = self.data[objs[0], :]
        for obj in objs[1:]:
            attrs_ind = np.logical_and(attrs_ind, self.data[obj, :])
        attrs_cl = self.attributes[np.asarray(attrs_ind).squeeze()]

        return objs, attrs_cl

    def calculate_lattice(self, num_level=5):
        self.lattice = {}

        if num_level <= 0:
            return self.lattice

        num_concepts = 0

        attrs = []

        while True:
            min_elem = 0
            if len(attrs) == num_level:
                min_elem = attrs[0]

            for attr in range(min_elem, self.num_attr):
                if len(attrs) > 0 and attr == attrs[0]:
                    attrs.pop(0)
                else:
                    attrs.insert(0, attr)

                    objs, attrs_cl = self.calculate_close(attrs)

                    if len(attrs_cl) > num_level:
                        attrs.pop(0)
                        continue

                    if len(attrs) + np.count_nonzero(attrs_cl < attr) < len(attrs_cl):
                        attrs.pop(0)
                        continue
                    else:
                        attrs = attrs_cl.tolist()
                        if len(objs) > 0 and len(attrs_cl) > 0 and len(objs) < self.num_obj:
                            self.lattice[num_concepts] = {'extent': objs, 'intent':attrs_cl}
                            num_concepts += 1
                        break

            if len(attrs) == 0:
                break

        return self.lattice

    def calculate_disjunctive_lattice(self, num_level=5):
        self.inverse_matrix()
        self.calculate_lattice(num_level)
        self.inverse_matrix()
        for i, concept in self.lattice.items():
            concept['extent'] = np.setdiff1d(np.arange(self.num_obj), concept['extent'])
        return self.lattice

    def save_lattice(self, file_name='data//lattice.txt'):
        fout = open(file_name, 'w')
        for index, concept in self.lattice.items():
            fout.writelines('%d;%s;%s\n' % (
                index,
                ','.join([str(d) for d in concept['extent']]),
                ','.join([str(d) for d in concept['intent']])
            ))
        fout.close()

    def load_lattice(self, file_name='data//lattice.txt'):
        self.lattice = {}
        finp = open(file_name, 'r')
        for row in finp:
            row_lex = row.strip('\n').split(';')
            self.lattice[int(row_lex[0])] = {
                'extent': np.array([int(s) for s in row_lex[1].split(',')]),
                'intent': np.array([int(s) for s in row_lex[2].split(',')])
            }
        finp.close()
        return self.lattice


class Diagram:

    def __init__(self, lattice, num_attributes, X):
        self.child_concepts = {}
        self.lattice = lattice
        self.num_attributes = num_attributes
        self.X = X

    def calculate_child_concepts(self):
        attribute_concept = np.ones((len(self.lattice), self.num_attributes), dtype='bool')

        self.child_concepts = {}

        for concept_index, concept in self.lattice.items():
            inverse_indices = np.ones(self.num_attributes, dtype='bool')
            inverse_indices[concept['intent']] = False
            self.child_concepts[concept_index] = \
                np.nonzero(attribute_concept[:concept_index, inverse_indices].all(axis=1))[0]
            attribute_concept[concept_index, concept['intent']] = False

    def calculate_probability(self, i, j_list, classes=None, type='concept'):
        if type == 'concept':
            return [float(len(self.lattice[j]['extent'])) / len(self.lattice[i]['extent']) for j in j_list]
        elif type == 'class':
            class_frequency = classes[self.lattice[i]['extent'], :].sum(axis=0).astype(dtype='float')
            return [class_frequency[j] / len(self.lattice[i]['extent']) for j in j_list]
        elif type == 'attribute':
            return [len(self.lattice[j]['extent']) / float(self.X[:, i].sum()) for j in j_list]
        else:
            print('Smth goes wrong in calculate_probability')

    def calculate_disjunctive_probability(self, i, j_list, classes=None, type='concept'):
        if type == 'concept':
            return [len(self.lattice[i]['extent']) / float(len(self.lattice[j]['extent'])) for j in j_list]
        elif type == 'class':
            class_frequency = classes[self.lattice[i]['extent'], :].sum(axis=0).astype(dtype='float')
            return [class_frequency[j] / len(self.lattice[i]['extent']) for j in j_list]
        elif type == 'attribute':
            return [float(self.X[:, i].sum()) / len(self.lattice[j]['extent']) for j in j_list]
        else:
            print('Smth goes wrong in calculate_disjunctive_probability')

    def calculate_quetle(self, i, j_list, classes, type='concept'):
        if type == 'concept':
            prob_i = float(classes[self.lattice[i]['extent']].sum()) / len(self.lattice[i]['extent'])
            return [float(classes[self.lattice[j]['extent']].sum()) / len(self.lattice[j]['extent']) / prob_i - 1
                    for j in j_list]
        elif type == 'class':
            prob_i = float(classes[self.lattice[i]['extent']].sum()) / len(self.lattice[i]['extent'])
            return [1 / prob_i - 1
                    for j in j_list]
        elif type == 'attribute':
            prob_i = float(classes[self.X[:, i] == 1].sum()) / np.count_nonzero(self.X[:, i] == 1)
            return [float(classes[self.lattice[j]['extent']].sum()) / len(self.lattice[j]['extent']) / prob_i - 1
                    for j in j_list]
        else:
            print('Smth goes wrong in calculate_quetle')

    def calculate_normalized_extent(self, i, j_list, classes=None, type='concept'):
        if type == 'concept':
            total_extent_sum = float(sum([len(self.lattice[j]['extent']) for j in j_list]))
            return [len(self.lattice[j]['extent']) / total_extent_sum for j in j_list]
        elif type == 'class':
            class_frequency = classes[self.lattice[i]['extent'], :].sum(axis=0).astype(dtype='float')
            return [class_frequency[j] / len(self.lattice[i]['extent']) for j in j_list]
        elif type == 'attribute':
            return [len(self.lattice[j]['extent']) / float(self.X[:, i].sum()) for j in j_list]
        else:
            print('Smth goes wrong in calculate_normalized_extent')

    def select_pure(self, classes, max_level=None, num_pure=10, min_support=10):
        concept_indices = []

        if max_level is None:
            purity_dict = {}
            for index, concept in self.lattice.items():
                if len(concept['extent']) >= min_support:
                    class_frequency = classes[concept['extent'], :].sum(axis=0)
                    purity_dict[index] = class_frequency.max() / class_frequency.sum()

            concept_indices = sorted(purity_dict.keys(), key = lambda x: purity_dict[x], reverse=True)[:num_pure]
            print(np.mean([purity_dict[index] for index in concept_indices]))
        else:
            purity_dict = {}
            for index, concept in self.lattice.items():
                if len(concept['extent']) >= min_support and len(concept['intent']) == max_level:
                    class_frequency = classes[concept['extent'], :].sum(axis=0)
                    purity_dict[index] = class_frequency.max() / class_frequency.sum()

            concept_indices_last = sorted(purity_dict.keys(), key=lambda x: purity_dict[x], reverse=True)[:num_pure]
            concept_indices = copy(concept_indices_last)
            print(np.mean([purity_dict[index] for index in concept_indices]))
            for index in concept_indices_last:
                concept_indices.extend(self.child_concepts[index].tolist())

        concept_indices = np.unique(concept_indices)
        concept_indices.sort()
        return concept_indices

    def select_accurate(self, classes, max_level=None, num_accurate=10, min_support=10):
        concept_indices = []

        if max_level is None:
            accuracy_dict = {}
            for index, concept in self.lattice.items():
                if len(concept['extent']) >= min_support:
                    bool_indices = np.ones(classes.shape[0], dtype='int')
                    bool_indices[concept['extent']] = 0
                    not_extent_indices = np.nonzero(bool_indices)[0]
                    class_frequency = classes[concept['extent'], :].sum(axis=0)
                    top_class = np.argmax(class_frequency)
                    tp = class_frequency[top_class]
                    fp = len(concept['extent']) - tp
                    fn = classes[not_extent_indices, top_class].sum()
                    tn = classes.shape[0] - tp - fp - fn

                    accuracy_dict[index] = float(tn+tp) / (tn+tp+fp+fn)

            concept_indices = sorted(accuracy_dict.keys(), key = lambda x: accuracy_dict[x], reverse=True)[:num_accurate]
            print(np.mean([accuracy_dict[index] for index in concept_indices]))
        else:
            accuracy_dict = {}
            for index, concept in self.lattice.items():
                if len(concept['extent']) >= min_support and len(concept['intent']) == max_level:
                    bool_indices = np.ones(classes.shape[0], dtype='int')
                    bool_indices[concept['extent']] = 0
                    not_extent_indices = np.nonzero(bool_indices)[0]
                    class_frequency = classes[concept['extent'], :].sum(axis=0)
                    top_class = np.argmax(class_frequency)
                    tp = class_frequency[top_class]
                    fp = len(concept['extent']) - tp
                    fn = classes[not_extent_indices, top_class].sum()
                    tn = classes.shape[0] - tp - fp - fn

                    accuracy_dict[index] = float(tn + tp) / (tn + tp + fp + fn)

            concept_indices_last = sorted(accuracy_dict.keys(), key=lambda x: accuracy_dict[x], reverse=True)[:num_accurate]
            concept_indices = concept_indices_last.copy()
            print(np.mean([accuracy_dict[index] for index in concept_indices]))
            for index in concept_indices_last:
                concept_indices.extend(self.child_concepts[index].tolist())

        concept_indices = np.unique(concept_indices)
        concept_indices.sort()
        return concept_indices

    def select_f_measure(self, classes, max_level=None, num_best=10, min_support=10):
        concept_indices = []

        if max_level is None:
            f_dict = {}
            for index, concept in self.lattice.items():
                if len(concept['extent']) >= min_support:
                    bool_indices = np.ones(classes.shape[0], dtype='int')
                    bool_indices[concept['extent']] = 0
                    not_extent_indices = np.nonzero(bool_indices)[0]
                    class_frequency = classes[concept['extent'], :].sum(axis=0)
                    top_class = np.argmax(class_frequency)
                    tp = class_frequency[top_class]
                    fp = len(concept['extent']) - tp
                    fn = classes[not_extent_indices, top_class].sum()
                    tn = classes.shape[0] - tp - fp - fn

                    f_dict[index] = 2 / (2 + float(fp) / tp + float(tn) / tp)

            concept_indices = sorted(f_dict.keys(), key = lambda x: f_dict[x], reverse=True)[:num_best]
            print(np.mean([f_dict[index] for index in concept_indices]))
        else:
            f_dict = {}
            for index, concept in self.lattice.items():
                if len(concept['extent']) >= min_support and len(concept['intent']) == max_level:
                    bool_indices = np.ones(classes.shape[0], dtype='int')
                    bool_indices[concept['extent']] = 0
                    not_extent_indices = np.nonzero(bool_indices)[0]
                    class_frequency = classes[concept['extent'], :].sum(axis=0)
                    top_class = np.argmax(class_frequency)
                    tp = class_frequency[top_class]
                    fp = len(concept['extent']) - tp
                    fn = classes[not_extent_indices, top_class].sum()
                    tn = classes.shape[0] - tp - fp - fn

                    f_dict[index] = 2 / (2 + float(fp) / tp + float(tn) / tp)

            concept_indices_last = sorted(f_dict.keys(), key=lambda x: f_dict[x], reverse=True)[:num_best]
            concept_indices = concept_indices_last.copy()
            print(np.mean([f_dict[index] for index in concept_indices]))
            for index in concept_indices_last:
                concept_indices.extend(self.child_concepts[index].tolist())

        concept_indices = np.unique(concept_indices)
        concept_indices.sort()
        return concept_indices

    def diagram_simple(self, num_levels=None, concept_indices=None, w_function=None):
        num_concepts = len(self.lattice)

        if concept_indices is None:
            concept_indices = np.arange(num_concepts)

        max_level = len(self.lattice[num_concepts-1]['intent'])

        if num_levels is None:
            num_levels = max_level

        if max_level > num_levels:
            concepts_on_each_level = [{} for i in range(num_levels)]

            for index in concept_indices:
                level = len(self.lattice[index]['intent'])
                if level <= num_levels:
                    concepts_on_each_level[level-1][index] = len(concepts_on_each_level[level-1])
        else:
            concepts_on_each_level = [{} for i in range(max_level)]

            for index in concept_indices:
                level = len(self.lattice[index]['intent'])
                concepts_on_each_level[level-1][index] = len(concepts_on_each_level[level-1])

            num_levels = max_level

        diagram = []
        W = []
        diagram.append(np.zeros((len(concepts_on_each_level[1]), self.num_attributes)))
        W.append(np.random.normal(scale=0.00001, size=(len(concepts_on_each_level[1]), self.num_attributes)))
        for j in concepts_on_each_level[1].keys():
            diagram[-1][concepts_on_each_level[1][j], self.lattice[j]['intent']] = 1
            if w_function is not None:
                for i in self.lattice[j]['intent']:
                    W[-1][concepts_on_each_level[1][j], i] = w_function(i, [j], type='attribute')[0]

        for i in range(2, num_levels):
            diagram.append(np.zeros((len(concepts_on_each_level[i]), len(concepts_on_each_level[i-1]))))
            W.append(np.random.normal(scale=0.00001, size=(len(concepts_on_each_level[i]), len(concepts_on_each_level[i-1]))))
            for j in concepts_on_each_level[i].keys():
                for k in self.child_concepts[j]:
                    try:
                        diagram[-1][concepts_on_each_level[i][j], concepts_on_each_level[i-1][k]] = 1
                        if w_function is not None:
                            W[-1][concepts_on_each_level[i][j], concepts_on_each_level[i-1][k]] = w_function(k, [j], type='concept')[0]
                    except:
                        pass

        return diagram, W

    def diagram_one_layer(self, num_levels=None, concept_indices=None):
        num_concepts = len(self.lattice)

        if concept_indices is None:
            concept_indices = np.arange(num_concepts)

        max_level = len(self.lattice[num_concepts-1]['intent'])

        if num_levels is None:
            num_levels = max_level

        if max_level > num_levels:
            concepts_on_each_level = [{} for i in range(2)]

            for index in concept_indices:
                level = len(self.lattice[index]['intent'])
                if level <= num_levels:
                    concepts_on_each_level[min(level-1, 1)][index] = len(concepts_on_each_level[min(level-1, 1)])
        else:
            concepts_on_each_level = [{} for i in range(2)]

            for index in concept_indices:
                level = len(self.lattice[index]['intent'])
                concepts_on_each_level[min(level - 1, 1)][index] = len(concepts_on_each_level[min(level - 1, 1)])

            num_levels = max_level

        diagram = []
        diagram.append(np.zeros((len(concepts_on_each_level[1]), self.num_attributes)))
        for j in concepts_on_each_level[1].keys():
            diagram[-1][concepts_on_each_level[1][j], self.lattice[j]['intent']] = 1

        return diagram

    def save_child_concepts(self, file_name='data//child_concepts.txt'):
        fout = open(file_name, 'w')
        for index, children in self.child_concepts.items():
            fout.writelines('%d;%s\n' % (
                index,
                ','.join([str(d) for d in children])
            ))
        fout.close()

    def load_child_concepts(self, file_name='data//child_concepts.txt'):
        self.child_concepts = {}
        finp = open(file_name, 'r')
        for row in finp:
            row_lex = row.strip('\n').split(';')
            if row_lex[1] == '':
                self.child_concepts[int(row_lex[0])] = np.array([])
            else:
                self.child_concepts[int(row_lex[0])] = np.array([int(s) for s in row_lex[1].split(',')])
        finp.close()
