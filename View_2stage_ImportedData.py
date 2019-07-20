import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('abnormal_beats/107_V1.csv', delimiter=',')
print(data.shape)
# [0, 1, 2, 3, 99, 200, 502, 2428, 2443]
for beatid in range(1000, 1150):
    try:
        times = np.arange(187, dtype='float')
        beat = data[beatid][:-1]
        anno = data[beatid][-1]
        plt.figure(figsize=(10, 5))
        print(anno)
        if anno == 0.0:
            plt.plot(times, beat, 'b', 'normal')
            print("this is normal")
        elif anno == 1.0:
            plt.plot(times, beat, 'r', 'LBBB')
            print("this is Left Bundle Branch")
        elif anno == 2.0:
            plt.plot(times, beat, 'r', 'RBBB')
            print("this is Right Bundle Branch")
        elif anno == 3.0:
            plt.plot(times, beat, 'r', 'BBB Beat')
            print("this is Bundle Branch Block Beat")
        elif anno == 4.0:
            plt.plot(times, beat, 'r', 'AF')
            print("this is Atrial Fibrillation")
        elif anno == 5.0:
            plt.plot(times, beat, 'r', 'Abberrated APB')
            print("this is Aberrated atrial premature beat")
        elif anno == 6.0:
            plt.plot(times, beat, 'r', 'Nodal PB')
            print("this is Nodal (junctional) premature beat")
        elif anno == 7.0:
            plt.plot(times, beat, 'r', 'SP, Supraventricular')
            print("this is Supraventricular premature or ectopic beat (atrial or nodal)")
        elif anno == 8.0:
            plt.plot(times, beat, 'r', 'PVC')
            print("this is Premature ventricular contraction")
        elif anno == 9.0:
            plt.plot(times, beat, 'r', 'RT PVC')
            print("this is R-on-T premature ventricular contraction")
        elif anno == 10.0:
            plt.plot(times, beat, 'r', 'Fusion of V and N')
            print("this is Fusion of ventricular and normal beat")
        elif anno == 11.0:
            plt.plot(times, beat, 'r')
            print("this is Atrial escape beat")
        elif anno == 12.0:
            plt.plot(times, beat, 'r')
            print("this is Nodal (junctional) escape beat")
        elif anno == 13.0:
            plt.plot(times, beat, 'r')
            print("this is Supraventricular escape beat (atrial or nodal)")
        elif anno == 14.0:
            plt.plot(times, beat, 'r')
            print("this is Ventricular escape beat")
        elif anno == 15.0:
            plt.plot(times, beat, 'r')
            print("this is Paced beat")
        elif anno == 16.0:
            plt.plot(times, beat, 'r')
            print("this is Fusion of paced and normal beat")
        elif anno == 17.0:
            plt.plot(times, beat, 'r')
            print("this is Unclassifiable beat")
        elif anno == 18.0:
            plt.plot(times, beat, 'r')
            print("this is Beat not classified during learning")
        plt.xlabel('Time [s]')
        plt.ylabel('beat' + str(beatid) + "type" + str(anno))
        plt.show()
    except IndexError:
        print("File is empty")

