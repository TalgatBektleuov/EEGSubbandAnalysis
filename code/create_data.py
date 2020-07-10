# Copyright (c) 2020 Herman Tarasau
# # Pre-processing
# coding: utf-8

import yaml
import mne
import numpy as np
from scipy.signal import stft
import pickle
from pathlib import Path


# Function to calculate IAF
def IAF(age):
    return 11.95 - 0.053 * age
# Implementing function to calculate ERD
def ERD(f, cal):
    return 100 * (cal - f) / cal
electrodes = {"central": ['EEG F3-Cz', 'EEG F4-Cz', 'EEG Fz-Cz', 'EEG C3-Cz', 'EEG C4-Cz', 'EEG P3-Cz', 'EEG P4-Cz',
                          'EEG Pz-Cz'],
              "frontal": ['EEG Fp1-Cz', 'EEG Fp2-Cz', 'EEG F7-Cz', 'EEG F3-Cz', 'EEG Fz-Cz', 'EEG F4-Cz', 'EEG F8-Cz'],
              "temporal": ['EEG T3-Cz', 'EEG T4-Cz', 'EEG T5-Cz', 'EEG T6-Cz']}
WAVES = ["L1A", "L2A", "UA", "Th", "Beta"]
def remove_outliers(erd):
    quant25 = np.quantile(erd, 0.25)
    quant75 = np.quantile(erd, 0.75)
    erd[(erd < quant25) | (erd > quant75)] = np.median(erd)
    return erd
CONFIG = yaml.load(open("subjects.yaml", "r"))

load_dir = Path(__file__).parent.parent / "data"
obg_dir = Path(__file__).parent.parent / "obj_dumps"
#file names
cal_name = "NabilaHosny.EyesClosed.Music.edf"
exp_name = "NabilaHosny.question2.Music.edf"

raw_cal = mne.io.read_raw_edf(cal_name, preload=True)
raw_exp = mne.io.read_raw_edf(exp_name, preload=True)
sfreq = raw_exp.info['sfreq']

# Data from specific channels
eyes = raw_cal.copy().pick_channels(ch_names=electrodes["central"])
experiment = raw_exp.copy().pick_channels(ch_names=electrodes["central"])

# Filtering AC line noise with notch filter

eyes_filtered_data = mne.filter.notch_filter(x=eyes.get_data(), Fs=sfreq, freqs=[50, 100])
experiment_filtered_data = mne.filter.notch_filter(x=experiment.get_data(), Fs=sfreq, freqs=[50, 100])
# eyes_filtered_data = eyes.get_data()
# experiment_filtered_data = experiment.get_data()

# Preparing data for plotting
eyes_filtered = mne.io.RawArray(data=eyes_filtered_data,
                                info=mne.create_info(ch_names=electrodes["central"], sfreq=sfreq))
experiment_filtered = mne.io.RawArray(data=experiment_filtered_data,
                                      info=mne.create_info(ch_names=electrodes["central"], sfreq=sfreq))

IAF_p = IAF(CONFIG["subjects"]["subject1"])

# Getting L1A, L2A, UA, Theta waves from eyes closed using FIR filtering. Also we take mean signal from all
# channels

eyes_sub_bands = {
    'L1A': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 4,
                                  h_freq=IAF_p - 2, sfreq=sfreq, method="fir"),
    'L2A': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 2,
                                  h_freq=IAF_p, sfreq=sfreq, method="fir"),
    'UA': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p,
                                 h_freq=IAF_p + 2, sfreq=sfreq, method="fir"),
    'Th': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 6,
                                 h_freq=IAF_p - 4, sfreq=sfreq, method="fir"),
    'Beta': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0),
                                   l_freq=IAF_p + 2,
                                   h_freq=30, sfreq=sfreq, method="fir")}

# Getting L1A, L2A, UA, Theta waves from experiment data using FIR filtering. Also we take mean signal from all
# channels
experiment_sub_bands = {"L1A": mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                      l_freq=IAF_p - 4, h_freq=IAF_p - 2, sfreq=sfreq,
                                                      method="fir"),
                        "L2A": mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                      l_freq=IAF_p - 2, h_freq=IAF_p, sfreq=sfreq,
                                                      method="fir"),
                        "UA": mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                     l_freq=IAF_p,
                                                     h_freq=IAF_p + 2, sfreq=sfreq, method="fir"),
                        "Th": mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                     l_freq=IAF_p - 6, h_freq=IAF_p - 4, sfreq=sfreq,
                                                     method="fir"),
                        "Beta": mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                       l_freq=IAF_p + 2, h_freq=30, sfreq=sfreq,
                                                       method="fir")}
# Calculating calibration values. Consider mean value of all channels. Va;ue are given in microvolts
eyes_sub_bands_ready={}
experiment_sub_bands_ready={}



for band in WAVES:
    eyes_sub_bands_ready[band] = np.mean(eyes_sub_bands[band]) * np.power(10, 6)
    experiment_sub_bands_ready[band] = np.mean(experiment_sub_bands[band]) * np.power(10, 6)

print("Experiment's power of sub-bands")
print(experiment_sub_bands_ready)

print("Resting power of sub-bands")
print(eyes_sub_bands_ready)

erd_mean_ready={}
for band in WAVES:
    erd_mean_ready[band]=ERD(experiment_sub_bands_ready[band],eyes_sub_bands_ready[band])
    
print("Total ERD values")
print(erd_mean_ready)

# # Performing STFT transform on experiment data for each sub-band. Window size is given in samples
# window = sfreq * 2
# fft = {}
#
# for band in WAVES:
#     fft[band] = stft(x=experiment_sub_bands[band], fs=sfreq, window=('kaiser', window), nperseg=1000)
#
# erd = np.vectorize(ERD)
# # Calculating ERD for experiment
# erd_mean = {}
#
# for band in fft:
#     curr_erd = erd(fft[band][2], calibration_values[band])
#     erd_mean[band] = remove_outliers(np.real(np.mean(curr_erd, axis=0)))
#
# # Adding clean Beta and UA energy ratio
# erd_mean["ABratio"] = remove_outliers(np.real(np.power(experiment_sub_bands["UA"] / experiment_sub_bands["Beta"], 2)))


