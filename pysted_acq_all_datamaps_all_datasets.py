import os
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pysted import base


action_spaces_new_photobleaching = {
    "pdt": {"low": 10.0e-6, "high": 150.0e-6},
    "p_ex": {"low": 0., "high": 42.2222e-6},
    "p_sted": {"low": 0., "high": 100 * 1.7681e-3}
}

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

background = 500   # TODO: find apporiate value
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

quality_th = 0.7
# datamaps_dir_list = [
#         f"./data/actin_3/all_quality_{quality_th}/datamaps_processed",
#         f"./data/camkii/CaMKII_Neuron/all_quality_{quality_th}/datamaps_processed",
#         f"./data/lifeact/LifeAct_Neuron/all_quality_{quality_th}/datamaps_processed",
#         f"./data/psd/PSD95_Neuron/all_quality_{quality_th}/datamaps_processed",
#         f"./data/tubulin/all_quality_{quality_th}/datamaps_processed",
# ]
datamaps_dir_list = [f"./data/dataset_bassoon/patches_datamaps"]

for datamaps_dir in datamaps_dir_list:
    sted_acqs_save_path = datamaps_dir + "/sted_acqs"
    if not os.path.exists(sted_acqs_save_path):
        os.mkdir(sted_acqs_save_path)

    confocal_acqs_save_path = datamaps_dir + "/confocal_acqs"
    if not os.path.exists(confocal_acqs_save_path):
        os.mkdir(confocal_acqs_save_path)

    n_datamaps = len([
        name for name in os.listdir(datamaps_dir)
        if os.path.isfile(os.path.join(datamaps_dir, name)) and name.split(".")[-1] == "npy"
    ])
    print(f"Doing acqs on datamaps at {datamaps_dir}")
    for i in tqdm.tqdm(range(n_datamaps)):
        molec_disp = np.array(500 * np.load(datamaps_dir + f"/{i}.npy"), dtype=int)

        datamap = base.Datamap(molec_disp, pixelsize)
        datamap.set_roi(i_ex, "max")

        acq_params_sted = {
            "pdt": action_spaces_new_photobleaching["pdt"]["low"],
            "p_ex": 0.35 * action_spaces_new_photobleaching["p_ex"]["high"],
            "p_sted": 0.2 * action_spaces_new_photobleaching["p_sted"]["high"]
        }

        acq_params_confoc = {
            "pdt": action_spaces_new_photobleaching["pdt"]["low"],
            "p_ex": 0.3 * action_spaces_new_photobleaching["p_ex"]["high"],
            "p_sted": 0.0 * action_spaces_new_photobleaching["p_sted"]["high"]
        }

        acq_sted, _, _ = microscope.get_signal_and_bleach(
            datamap, datamap.pixelsize, **acq_params_sted, bleach=False, update=False
        )

        acq_confoc, _, _ = microscope.get_signal_and_bleach(
            datamap, datamap.pixelsize, **acq_params_confoc, bleach=False, update=False
        )

        np.save(sted_acqs_save_path + f"/{i}", acq_sted)
        np.save(confocal_acqs_save_path + f"/{i}", acq_confoc)
