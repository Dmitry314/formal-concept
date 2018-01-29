import numpy as np
from data_loader import *
import sklearn.neighbors as nn

# print(sklearn.__path__)

model = nn.KNeighborsClassifier(n_neighbors=1)
X, y, object_labels, attribute_labels = get_titanic()[:4]

y_cl = one_hot(y, n_classes=7)

result = []
result_f_measure = []

for i in range(20):
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y_cl, tp=0.6, vp=0.2)

    model.fit(X_train, y_train)

    y_pr = model.predict(X_test)

    result.append((y_test.shape[0]-np.count_nonzero(y_pr-y_test)/2) / y_test.shape[0])
    tp = np.multiply(y_pr, y_test).sum(axis=0).astype(dtype='float')
    t = y_test.sum(axis=0)+np.ones(y_test.shape[1])*0.0001
    p = y_pr.sum(axis=0)+np.ones(y_test.shape[1])*0.0001

    result_f_measure.append(np.mean(2*np.multiply(tp / t, tp / p) / (tp / t + tp / p + np.ones(y_test.shape[1])*0.0001)))
print(result)
print(np.mean(result))
print(result_f_measure)
print(np.mean(result_f_measure))