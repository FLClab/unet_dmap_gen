import os
import numpy as np
import pandas as pd
from pysted import base
from matplotlib import pyplot as plt


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

pixelsize = 20e-9
# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(635e-9)
laser_sted = base.DonutBeam(750e-9, zero_residual=0)
detector = base.Detector(noise=True, background=5000000)   # TODO : add du detector noise
objective = base.Objective()
objective.transmission[690] = 0.85
fluo = base.Fluorescence(**FLUO_NEW_PHOTOBLEACHING)
microscope = base.Microscope(
    laser_ex, laser_sted, detector, objective, fluo, load_cache=False
)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=False)

action_spaces = {
    "pdt": {"low": 10.0e-6, "high": 150.0e-6},
    "p_ex": {"low": 0., "high": 42.2222e-6},
    "p_sted": {"low": 0., "high": 100 * 1.7681e-3}
}

acqs_save_dir = "./data/beads_acq"
if not os.path.exists(acqs_save_dir):
    os.mkdir(acqs_save_dir)

n_acqs = 1000
pdt_list, p_ex_list, p_sted_list, n_molecs_beads_list = [], [], [], []
for i in range(n_acqs):
    acq_params = {
        "pdt": action_spaces["pdt"]["low"],
        "p_ex": np.random.uniform(0.2, 0.5) * action_spaces["p_ex"]["high"],
        "p_sted": np.random.uniform(0.1, 0.3) * action_spaces["p_sted"]["high"]
    }

    beads_pos = np.random.randint(0, 64, size=(20, 2))
    molecs_disp = np.zeros((64, 64))
    n_molecs = np.random.randint(75, 125)
    molecs_disp[beads_pos[:, 0], beads_pos[:, 1]] += n_molecs
    datamap = base.Datamap(molecs_disp, pixelsize)
    datamap.set_roi(i_ex, "max")

    pdt_list.append(acq_params["pdt"])
    p_ex_list.append(acq_params["p_ex"])
    p_sted_list.append(acq_params["p_sted"])
    n_molecs_beads_list.append(n_molecs)

    acq, _, _ = microscope.get_signal_and_bleach(datamap, datamap.pixelsize,
                                                 **acq_params,
                                                 bleach=False, update=False)

    np.save(acqs_save_dir + f"/{i}", acq)

data = {
    "pdt": pdt_list,
    "p_ex": p_ex_list,
    "p_sted": p_sted_list,
    "n_molecs per bead": n_molecs_beads_list
}
dataframe = pd.DataFrame(data=data)
dataframe.to_csv(acqs_save_dir + "/info.csv")
