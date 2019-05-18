import numpy as np
from matplotlib import pyplot as plt


data = np.loadtxt('data_ecg/233_V1.csv', delimiter=',')
print(data.shape)

for beatid in [0, 1, 2, 3, 99, 200, 502, 2428, 2443]:
    try:
        times = np.arange(187, dtype='float')
        beat = data[beatid][:-1]
        anno = data[beatid][-1]
        plt.figure(figsize=(20, 5))
        if anno == 0.0:
            plt.plot(times, beat, 'b')
            print("this is normal")
        else:
            plt.plot(times, beat, 'r')
            print("this is abnormal")
        plt.xlabel('Time [s]')
        plt.ylabel('beat' + str(beatid) + "type" + str(anno))
        plt.show()
    except IndexError:
        print("File is empty")

