from fca import FCA, Diagram
import numpy as np
from data_loader import *
import time
from nn_layers import DeepSparseNet, DeepDenseNet
from lasagne import nonlinearities
import networkx as nx
result_accuracy = []
result_f_measure = []
result_num_epoch = []

for i in range(20):

    #X, y, object_labels, attribute_labels = get_zoo()[:4]

    X, y, object_labels, attribute_labels = get_titanic(file_name='data_sets//titanic.txt')[:4]
    # get_random(1000, 15, frequency=0.8)[:4]
    # get_titanic(file_name='data_sets//titanic.txt')[:4]
    # get_mammographic_masses()[:4]
    # get_breast_cancer()[:4]
    # get_seismic_bumps()[:4]
    # get_car_evaluation()[:4]
    # get_zoo()[:4]
    y_cl = one_hot(y)

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y_cl, tp=0.6, vp=0.2)

  
    fca = FCA(X_train)
    l = fca.calculate_disjunctive_lattice(num_level=5)
    # l = fca.calculate_lattice(num_level=4)
    fca.save_lattice()
    l = fca.load_lattice()

    diag = Diagram(lattice=l, num_attributes=X_train.shape[1], X=X)
    diag.calculate_child_concepts()
    diag.save_child_concepts()
    diag.load_child_concepts()
    
    
    print(diag.child_concepts)
    import networkx as nx
    G=nx.Graph()
    
    for i in diag.child_concepts.keys():
        for j in range(len(diag.child_concepts[i])):
            G.add_edge(i, diag.child_concepts[i][j])
            
    import matplotlib.pyplot as plt

    #nx.draw(G)
    # simple diagram # 3
    num_level = 3
    concept_indices = diag.select_pure(y_cl, num_level)
    # concept_indices = diag.select_f_measure(y_cl, num_level)
    simple_diagram, W = diag.diagram_simple(num_levels=num_level, concept_indices=concept_indices,
                                            w_function=diag.calculate_disjunctive_probability)
    # w_function=diag.calculate_probability)
    # w_function=diag.calculate_disjunctive_probability)

    """for i in range(len(simple_diagram)):
        print('C:\n', simple_diagram[i])
        print('W:\n', W[i])"""

    dsn = DeepSparseNet(simple_diagram, num_classes=y_train.shape[1], nonlinearity=nonlinearities.tanh, W=W)
    #ddn = DeepDenseNet(simple_diagram, num_classes=y_train.shape[1], nonlinearity=nonlinearities.tanh)

    for layer in dsn.layers:
        print('%s:' % layer.name, layer.output_shape)


    max_accuracy = 0
    best_params = None
    num_epoch_without_best = 50
    count_epoch = 0
    total_count_epoch = 0
    for epoch, loss in enumerate(ddn.fit(X.astype(dtype='float32'), y_cl.astype(dtype='float32'), n_epoches=400, batch_size=32)):
        count_epoch += 1
        total_count_epoch += 1

        val_acc = ddn.calculate_accuracy(X_val, y_val)
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            best_params = ddn.get_param_values()
            count_epoch = 0

        if count_epoch > num_epoch_without_best:
            break

        """
        if epoch % 20 > 0:
            continue
        print('--> %d' % epoch)
        print('    loss:', np.mean(loss))
        train_acc = dsn.calculate_accuracy(X_train, y_train)
        print('    train_acc:', train_acc)
        print('    val_acc:', val_acc)"""

    ddn.set_param_values(best_params)
    test_acc = ddn.calculate_accuracy(X_test, y_test)
    test_f = ddn.calculate_f_measure(X_test, y_test)
    print('RESULTS (%d):\n' % i)
    print(test_acc, test_f, total_count_epoch)
    result_accuracy.append(test_acc)
    result_f_measure.append(test_f)
    result_num_epoch.append(total_count_epoch)

print('result_accuracy')
print(result_accuracy)
print('result_f_measure')
print(result_f_measure)
print('result_num_epoch')
print(result_num_epoch)

