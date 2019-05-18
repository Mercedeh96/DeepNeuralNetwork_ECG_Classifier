import numpy as np
import wfdb as wf
from biosppy.signals import ecg
from scipy import signal

realbeats = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
             'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']

for i in range(100, 235):
    try:
        record = wf.rdsamp("Signals/" + str(i))
        annotation = wf.rdann("Signals/" + str(i), 'atr')
        print('    Sampling frequency used for this record:', record[1].get('fs'))
        print('    Shape of loaded data array:', record[0].shape)
        print('    Number of loaded annotations:', len(annotation.num))
        # Get the ECG values from the file.
        data = record[0].transpose()
        # Generate the classifications based on the annotations.
        # 0.0 = undetermined
        # 1.0 = normal
        # 2.0 = abnormal
        cat = np.array(annotation.symbol)
        rate = np.zeros_like(cat, dtype='float')
        for catid, catval in enumerate(cat):
            if catval == 'N':
                rate[catid] = 1.0  # Normal
                # print("this is normal")
            elif catval in realbeats:
                rate[catid] = 2.0  # Abnormal
                # print("this is abnormal")
        rates = np.zeros_like(data[0], dtype='float')
        rates[annotation.sample] = rate
        indices = np.arange(data[0].size, dtype='int')
        # Process each channel separately (2 per input file).
        for channelid, channel in enumerate(data):
            chname = record[1].get('sig_name')[channelid]
            print('    ECG channel type:', chname)
            # Find rpeaks in the ECG data. Most should match with
            # the annotations. The biosppy library is used to detect r peaks
            out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
            rpeaks = np.zeros_like(channel, dtype='float')
            rpeaks[out['rpeaks']] = 1.0

            beatstoremove = np.array([0])
            # Split into individual heartbeats. For each heartbeat
            # record, append classification (normal/abnormal).
            beats = np.split(channel, out['rpeaks'])
            for idx, idxval in enumerate(out['rpeaks']):
                firstround = idx == 0
                lastround = idx == len(beats) - 1

                # Skip first and last beat.
                if firstround or lastround:
                    continue

                # Get the classification value that is on
                # or near the position of the rpeak index.
                fromidx = 0 if idxval < 10 else idxval - 10
                toidx = idxval + 10
                catval = rates[fromidx:toidx].max()
                # print("fromidx is " + str(fromidx))
                # print("toidx is " + str(toidx))
                # print("catval is " + str(catval))

                # Skip beat if there is no classification. Here we are saying that when catval is 0
                # there is no classification so these beats with no classifications are being removed
                if catval == 0.0:
                    beatstoremove = np.append(beatstoremove, idx)
                    # print("beatstoremove is " + str(beatstoremove))
                    continue

                # Normal beat is now classified as 0.0 and abnormal is 1.0. because before we said normal is 1 and
                # abnormal is 2 so that's why the catval now changes to 0 and 1
                catval = catval - 1.0
                # print(catval)

                # Append some extra readings from next beat.
                beats[idx] = np.append(beats[idx], beats[idx + 1][:40])

                # Normalize the readings to a 0-1 range for ML purposes.
                beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()

                # Resample from 360Hz to 125Hz
                newsize = int((beats[idx].size * 125 / 360) + 0.5)
                beats[idx] = signal.resample(beats[idx], newsize)

                # Skipping records that are too long.
                if beats[idx].size > 187:
                    beatstoremove = np.append(beatstoremove, idx)
                    continue

                # Pad with zeroes.
                zerocount = 187 - beats[idx].size
                beats[idx] = np.pad(beats[idx], (0, zerocount), 'constant', constant_values=(0.0, 0.0))

                # Append the classification to the beat data.
                beats[idx] = np.append(beats[idx], catval)

            beatstoremove = np.append(beatstoremove, len(beats) - 1)

            # Remove first and last beats and the ones without classification.
            beats = np.delete(beats, beatstoremove)

            # Save to CSV file.
            savedata = np.array(list(beats[:]), dtype=np.float)
            outfn = 'data_ecg/' + str(i) + '_' + chname + '.csv'
            print('Generating ', outfn)
            with open(outfn, "wb") as fin:
                np.savetxt(fin, savedata, delimiter=",", fmt='%f')

    except FileNotFoundError:
        print("An exception occurred, file is not found for this record")
        continue
