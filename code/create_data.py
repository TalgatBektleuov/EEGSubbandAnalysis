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

CONFIG = yaml.load(open("subjects.yaml", "r"))

WAVES = ["L1A", "L2A", "UA", "Th"]


def remove_outliers(erd):
    quant25 = np.quantile(erd, 0.25)
    quant75 = np.quantile(erd, 0.75)
    erd[(erd < quant25) | (erd > quant75)] = np.median(erd)
    return erd


load_dir = Path(__file__).parent.parent / "data"
obg_dir = Path(__file__).parent.parent / "obj_dumps"
(obg_dir / "subject1").mkdir(parents=True, exist_ok=True)
curr_dir = load_dir / "subject1" / "music"

#file names
cal_name = "NabilaHosny.EyesClosed.Control.edf"
exp_name = "NabilaHosny.question1.Control.edf"

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
print("L1A is ")
print({'L1A': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                      l_freq=IAF_p - 4, h_freq=IAF_p - 2, sfreq=sfreq,
                                                      method="fir")})
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

eyes_sub_bands_ready={"L1A": np.mean(eyes_sub_bands.get("L1A", "NONE")),
                            "L2A": np.mean(eyes_sub_bands.get("L2A", "NONE")),
                            "UA": np.mean(eyes_sub_bands.get("UA", "NONE")),
                            "Theta": np.mean(eyes_sub_bands.get("Th", "NONE")),
                            "Beta": np.mean(eyes_sub_bands.get("Beta", "NONE")) }

experiment_sub_bands_ready={"L1A": np.mean(experiment_sub_bands.get("L1A", "NONE")),
                            "L2A": np.mean(experiment_sub_bands.get("L2A", "NONE")),
                            "UA": np.mean(experiment_sub_bands.get("UA", "NONE")),
                            "Theta": np.mean(experiment_sub_bands.get("Th", "NONE")),
                            "Beta": np.mean(experiment_sub_bands.get("Beta", "NONE")) }
print(eyes_sub_bands_ready)
print(experiment_sub_bands_ready)


