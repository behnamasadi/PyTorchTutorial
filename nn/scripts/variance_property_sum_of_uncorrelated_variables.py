import matplotlib.pyplot as plt
import numpy as np

x=[]
stds=[]
means=[]

num_cols=5

fig, axes=plt.subplots(nrows=1,ncols=num_cols)
number_of_ditributions=50
for i in np.arange(number_of_ditributions):
        rnd=np.random.randn(1000)
        x.append(rnd)
        means.append(np.mean(rnd))
        stds.append(np.std(rnd))
        if i%(np.round(number_of_ditributions/num_cols) )==0:
                X = np.sum(x, axis=0)
                print("std is :",np.std(X))
                print("mean is :", np.mean(X))
                idx=int(i /np.round(number_of_ditributions / num_cols))
                axes[idx].hist(X, 100, range=(-20, 20))

plt.show()
