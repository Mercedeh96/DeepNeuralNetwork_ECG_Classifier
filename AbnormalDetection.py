import numpy as np
import wfdb as wf
from biosppy.signals import ecg
from scipy import signal


realbeats = ['L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
             'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,
               114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
               203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220,
               221, 222, 223, 228, 230, 231, 232, 233, 234]


for i in RECORDS:
    try:
        record = wf.rdsamp("Signals/" + str(i))
        annotation = wf.rdann("Signals/" + str(i), 'atr')
        print('    Sampling frequency used for this record:', record[1].get('fs'))
        print('    Shape of loaded data array:', record[0].shape)
        print('    Number of loaded annotations:', len(annotation.num))

        data = record[0].transpose()  # transpose to be done before the signal can be processed using the biosppy
        cat = np.array(annotation.symbol)  # getting the annotation symbols for each reocrd
        rate = np.zeros_like(cat, dtype='float')

        # enumerating through the annotations,if N then normal if else then abnormal
        for cat_id, cat_val in enumerate(cat):
            if cat_val == 'N':
                rate[cat_id] = 1.0  # Normal seperated from the abnormal ones
                # print("this is normal")
            elif cat_val == "L":
                rate[cat_id] = 2.0  # Left Bundle Branch
            elif cat_val == "R":
                rate[cat_id] = 3.0  # Right Bundle Branch
            elif cat_val == "B":
                rate[cat_id] = 4.0  # Bundle Branch Block Beat
            elif cat_val == "A":
                rate[cat_id] = 5.0  # Atrial Fibrillation
            elif cat_val == "a":
                rate[cat_id] = 6.0  # Aberrated atrial premature beat
            elif cat_val == "J":
                rate[cat_id] = 7.0  # Nodal (junctional) premature beat
            elif cat_val == "S":
                rate[cat_id] = 8.0  # Supraventricular premature or ectopic beat (atrial or nodal)
            elif cat_val == "V":
                rate[cat_id] = 9.0  # Premature ventricular contraction
            elif cat_val == "r":
                rate[cat_id] = 10.0  # R-on-T premature ventricular contraction
            elif cat_val == "F":
                rate[cat_id] = 11.0  # Fusion of ventricular and normal beat
            elif cat_val == "e":
                rate[cat_id] = 12.0  # Atrial escape beat
            elif cat_val == "j":
                rate[cat_id] = 13.0  # Nodal (junctional) escape beat
            elif cat_val == "n":
                rate[cat_id] = 14.0  # Supraventricular escape beat (atrial or nodal)
            elif cat_val == "E":
                rate[cat_id] = 15.0  # Ventricular escape beat
            elif cat_val == "/":
                rate[cat_id] = 16.0  # Paced beat
            elif cat_val == "f":
                rate[cat_id] = 17.0  # Fusion of paced and normal beat
            elif cat_val == "Q":
                rate[cat_id] = 18.0  # Unclassifiable beat
            elif cat_val == "?":
                rate[cat_id] = 19.0  # Beat not classified during learning
        rates = np.zeros_like(data[0], dtype='float')
        rates[annotation.sample] = rate
        indices = np.arange(data[0].size, dtype='int')
        for notes in annotation.aux_note:
            if "(N" in notes:
                print("this is a normal rhythm")
            if "(SBR" in notes:
                print("this is Sinus bradycardia")
            if "(P" in notes:
                print("this is paced rhythm")
            if "(AB" in notes:
                print("this is Atrial bigeminy")
            if "(AFIB" in notes:
                print("this is Atrial fibrillation")
            if "(AFL" in notes:
                print("this is Atrial flutter")
            if "(B" in notes:
                print("this is Ventricular bigeminy")
            if "(BII" in notes:
                print("this is 2° heart block")
            if "(IVR" in notes:
                print("this is Idioventricular rhythm")
            if "(NOD" in notes:
                print("this is Nodal (A-V junctional) rhythm")
            if "(PREX" in notes:
                print("this is Pre-excitation (WPW)")
            if "(SVTA" in notes:
                print("this is Supraventricular tachyarrhythmia")
            if "(T" in notes:
                print("this is Ventricular trigeminy")
            if "(VFL" in notes:
                print("this is Ventricular flutter")
            if "(VT" in notes:
                print("this is Ventricular tachycardia")

        for channelid, channel in enumerate(data):
            chname = record[1].get('sig_name')[channelid]
            print('    ECG channel type:', chname)
            out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
            rpeaks = np.zeros_like(channel, dtype='float')
            rpeaks[out['rpeaks']] = 1.0

            beatstoremove = np.array([0])
            # Split into individual heartbeats. For each heartbeat
            # record, append classification
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
            outfn = 'abnormal_beats/' + str(i) + '_' + chname + '.csv'
            print('Generating ', outfn)
            with open(outfn, "wb") as fin:
                np.savetxt(fin, savedata, delimiter=",", fmt='%f')

    except FileNotFoundError:
        print("File not found for this record")
