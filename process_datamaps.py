import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


real_acqs_dir = "./data/actin/verif_all"
datamaps_dir = "./data/actin/verif_datamaps"
processed_datamaps_dir = "./data/actin/datamaps_processed"
fig_save_path = "./data/actin/lab_meeting_figs"

img_quality_dataframe = pd.read_csv(real_acqs_dir + f"/img_quality.csv")

if not os.path.exists(processed_datamaps_dir):
    os.mkdir(processed_datamaps_dir)

n_datamaps = len([name for name in os.listdir(datamaps_dir) if os.path.isfile(os.path.join(datamaps_dir, name))])
# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
thresholds = [0.25]
threshold_subdirs = [processed_datamaps_dir + f"/{threshold}" for threshold in thresholds]
for threshold_subdir in threshold_subdirs:
    if not os.path.exists(threshold_subdir):
        os.mkdir(threshold_subdir)

for i in range(n_datamaps):
    datamap = np.load(datamaps_dir + f"/{i}.npy")

    # processed_datamaps = []
    for threshold in thresholds:
        datamap_processed = np.where(datamap < threshold, 0, datamap)
        np.save(processed_datamaps_dir + f"/{threshold}/{i}", datamap_processed)
        # processed_datamaps.append(datamap_processed)

    # fig, axes = plt.subplots(1, len(thresholds) + 1)
    #
    # axes[0].imshow(datamap)
    # axes[0].set_title(f"Unprocessed datamap")
    # for j in range(len(thresholds)):
    #     axes[j + 1].imshow(processed_datamaps[j])
    #     axes[j + 1].set_title(f"threshold = {thresholds[j]}")
    # plt.show()


"""
# plot histogram of pixel values

n_datamaps = len([name for name in os.listdir(datamaps_dir) if os.path.isfile(os.path.join(datamaps_dir, name))])
n_bins = 20

datamap_histograms = np.zeros((n_datamaps, n_bins))
for i in range(n_datamaps):
    datamap = np.load(datamaps_dir + f"/{i}.npy")
    # real_acq = np.load(real_acqs_dir + f"/{i}")["arr_0"]
    datamap_flat = datamap.flatten()
    n, bins, _ = plt.hist(datamap_flat, bins=n_bins)
    datamap_histograms[i, :] = n
    plt.close()

datamap_hist_avg = np.mean(datamap_histograms, axis=0)
datamap_hist_std = np.std(datamap_histograms, axis=0)

plt.style.use('dark_background')
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.bar(bins[:-1], datamap_hist_avg, yerr=datamap_hist_std, align="edge", width=0.05,
       error_kw=dict(ecolor='white'))
ax.set_xlabel(f"Pixel value")
ax.set_ylabel(f"Number of pixel")

plt.savefig(fig_save_path + f"/datamap_histogram_values.png")
plt.savefig(fig_save_path + f"/datamap_histogram_values.jpg")
plt.savefig(fig_save_path + f"/datamap_histogram_values.pdf")
plt.show()
"""

