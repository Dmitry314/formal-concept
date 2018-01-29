from sklearn.naive_bayes import GaussianNB
from data_loader import *


X, y, object_labels, attribute_labels = get_zoo()[:4]

result = []
result_f_measure = []

for i in range(20):
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(X, y, tp=0.6, vp=0.2)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    print((y_pred == y_test).sum().astype('float') / y_pred.shape[0])

    print(np.unique())