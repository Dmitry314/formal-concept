import random
import numpy as np

ind = np.array(random.sample(range(5000), 2500))
x = np.zeros(5000, dtype='int')
x[ind] = 1

fout = open('temp_to_mail.txt', 'w')
for a in x:
    fout.writelines('%d\n' % a)
fout.close()

