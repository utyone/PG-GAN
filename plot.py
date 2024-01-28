from matplotlib import pyplot as plt
import numpy as np

for epoch in range(0,18000,100):
    data = np.loadtxt("results/gen_data_e{}.csv".format(epoch), delimiter=",")

    num = (int)(data.shape[0]/2)
    print(num, data.shape)

    fig = plt.figure()
    for i in range(num-1):
        plt.plot(data[2*i+2,:], data[2*i+1,:])
    #plt.show()
    fig.savefig("results/gen_{}.png".format(epoch))
    

