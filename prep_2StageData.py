from glob import glob
import numpy as np
from matplotlib import pyplot as plt

# read all the csv data files into memory
# alldata = np.empty(shape=[0, 188])
# print(alldata.shape)
beats_to_keep = np.empty(shape=[0, 188])
# print(beats_to_keep.shape)
paths = glob('abnormal_beats/*.csv')
for path in paths:
    print('Loading', path)
    csvrows = np.loadtxt(path, delimiter=',')
    times = np.arange(187, dtype='float')
    try:
        # print(csvrows.shape[0]) # number of rows
        # print(csvrows.shape[1]) # number of columns
        # for row in csvrows.shape[0]:
        #     for column in csvrows.shape[1]:
        #             anno = csvrows[row][-1]
        #             if anno != 0:
        #                 beats_to_keep = np.append(beats_to_keep, row, column)
        #                 print(beats_to_keep)

        # print(len(csvrows))
        for beatid in range(len(csvrows)):
            beat = csvrows[beatid][:-1]
            anno = csvrows[beatid][-1]
            # print(beat)
            # print(anno)
            # print(len(csvrows))
            if anno != 0.0:
                # print(beats_to_keep.shape)
                beats_to_keep = np.append(beats_to_keep, csvrows)
        #print(beats_to_keep)
    except IndexError:
        print("No file found")

# np.random.shuffle(beats_to_keep)
# totrows = len(beats_to_keep)
# trainrows = int((totrows * 3 / 5) + 0.5)  # 60%
# testrows = int((totrows * 1 / 5) + 0.5)  # 20%
# validaterows = totrows - trainrows - testrows  # 20%
# mark1 = trainrows
# mark2 = mark1 + testrows

# data is saved in three separate files, training (%60), testing(%20), validation(%20)
with open('New_2stage_train.csv', "wb") as fin:
    np.savetxt(fin, beats_to_keep[:mark1], delimiter=",", fmt='%f')

with open('New_2stage_test.csv', "wb") as fin:
    np.savetxt(fin, beats_to_keep[mark1:mark2], delimiter=",", fmt='%f')

with open('New_2stage_validate.csv', "wb") as fin:
    np.savetxt(fin, beats_to_keep[mark2:], delimiter=",", fmt='%f')

# # shuffle and separating the data
# # Randomly mix rows
# np.random.shuffle(alldata)
# totrows = len(alldata)
# trainrows = int((totrows * 3 / 5) + 0.5) # 60%
# testrows = int((totrows * 1 / 5) + 0.5) # 20%
# validaterows = totrows - trainrows - testrows # 20%
# mark1 = trainrows
# mark2 = mark1 + testrows
#
# # data is saved in three separate files, training (%60), testing(%20), validation(%20)
# with open('2stage_train.csv', "wb") as fin:
#     np.savetxt(fin, alldata[:mark1], delimiter=",", fmt='%f')
#
# with open('2stage_test.csv', "wb") as fin:
#     np.savetxt(fin, alldata[mark1:mark2], delimiter=",", fmt='%f')
#
# with open('2stage_validate.csv', "wb") as fin:
#     np.savetxt(fin, alldata[mark2:], delimiter=",", fmt='%f')
