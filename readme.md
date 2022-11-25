This repo contains code to train and run a UNet to generate pySTED Datamaps from real STED images.

A trained model is available VALERIA:flclab-public/unet_datamap_generation/20220726-095941_udaxihhe/best.pt

Model was trained on a combination of the following datasets from https://lvsn.github.io/optimnanoscopy/

Actin
Tubulin
PSD95
aCaMKII
LifeAct

using only the images with a quality score >= 0.7. Model takes as input 224x224 pixel images and outputs datamaps of the
same size.

The file run_generalist_model_datasets.py shows an example of how to run the model on the separate datasets.


