"""
build microscope
go through the folder with all the datamaps
load the datamap
process it
build datamap
acquire on datamap
save acquisition
"""

import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pysted import base
matplotlib.use('tkagg')

action_spaces_new_photobleaching = {
    "pdt": {"low": 10.0e-6, "high": 150.0e-6},
    "p_ex": {"low": 0., "high": 42.2222e-6},
    "p_sted": {"low": 0., "high": 100 * 1.7681e-3}
}

datamaps_dir = os.path.expanduser("~/Documents/a22_docs/micranet_dataset_test/datamaps_processed")
real_acqs_dir = os.path.expanduser("~/Documents/a22_docs/micranet_dataset_test")
n_datamaps = len([name for name in os.listdir(datamaps_dir) if os.path.isfile(os.path.join(datamaps_dir, name))])

FLUO_NEW_PHOTOBLEACHING = {   # ATTO647N
    # these are the params Anthony uses as of 06/06/22, with a molec count of ~100
    "lambda_": 690e-9, # Figure 1, Oracz2017
    "qy": 0.65,  # Product Information: ATTO 647N (ATTO-TEC GmbH)
    "sigma_abs": {
        635: 1.0e-20, #Table S3, Oracz2017
        750: 3.5e-25,  # (1 photon exc abs) Table S3, Oracz2017
    },
    "sigma_ste": {
        750: 4.8e-22, #Table S3, Oracz2017
    },
    "tau": 3.5e-9, #Table S3, Oracz2017
    "tau_vib": 1.0e-12, #t_vib, Table S3, Oracz2017
    "tau_tri": 25e-6, # pasted from egfp
    "k0": 0, #Table S3, Oracz2017
    "k1": 1.3e-15, #Table S3,  (changed seemingly wrong unit: 5.2 × 10−10 / (100**2)**1.4)
    "b":1.45, #Table S3, Oracz2017; original value of 1.4; value slightly higher to increase difficulty
    "triplet_dynamic_frac": 0, #Ignore the triplet dynamics by default
}

background = 1000000
pixelsize = 20e-9
# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(635e-9)
laser_sted = base.DonutBeam(750e-9, zero_residual=0)
detector = base.Detector(noise=True, background=background)
objective = base.Objective()
objective.transmission[690] = 0.85
fluo = base.Fluorescence(**FLUO_NEW_PHOTOBLEACHING)
microscope = base.Microscope(
    laser_ex, laser_sted, detector, objective, fluo, load_cache=False
)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=False)

acq_save_path = datamaps_dir + f"/acqs_bckgrnd_{background}"
if not os.path.exists(acq_save_path):
    os.mkdir(acq_save_path)

for i in range(n_datamaps):
    real_acq = np.load(real_acqs_dir + f"/{i}")["arr_0"]

    print(f"acq {i+1} of {n_datamaps}")
    molec_disp = np.array(1000 * np.load(datamaps_dir + f"/{i}.npy"), dtype=int)

    datamap = base.Datamap(molec_disp, pixelsize)
    datamap.set_roi(i_ex, "max")

    acq_params_sted = {
        "pdt": action_spaces_new_photobleaching["pdt"]["low"],
        "p_ex": 0.35 * action_spaces_new_photobleaching["p_ex"]["high"],
        "p_sted": 0.2 * action_spaces_new_photobleaching["p_sted"]["high"]
    }

    acq_sted, _, _ = microscope.get_signal_and_bleach(
        datamap, datamap.pixelsize, **acq_params_sted, bleach=False, update=False
    )

    normalized_acq = (acq_sted - np.min(acq_sted)) / (np.max(acq_sted) - np.min(acq_sted))

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(real_acq, cmap="hot")
    axes[0].set_title(f"REAL \n"
                      f"min = {np.min(real_acq)}, max = {np.max(real_acq)}")
    axes[1].imshow(normalized_acq, cmap="hot")
    axes[1].set_title(f"pySTED (normalized) \n"
                      f"min = {np.min(normalized_acq)}, max = {np.max(normalized_acq)}")

    plt.show()

    # np.save(acq_save_path + f"/{i}", acq_sted)





