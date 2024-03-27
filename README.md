# U-Net datamap

This is a U-Net implementation for datamap generation used in pySTED.

## Training

To train the model, run the following command:

```bash
python train.py
```

This assumes that the training and testing data is in the `results/data` directory. The training data should be in the `train_quality_0.7` subdirectory and the testing data should be in the `test_quality_0.7` subdirectory.

## Using the model

To use the model, run the following command:

```bash
python run_generalist_model_complete-images.py --folder "<NAMEOFFOLDER>"
```

Where `<NAMEOFFOLDER>` is the name of the folder containing the images to be processed. The images should be in the `results/datamaps/<NAMEOFFOLDER>` directory. All tifffiles in the folder will be recursively processed. Large images will be split into smaller images (224x224 pixels) and saved separately. The saved image name will include the name of the image and the coordinates of the image in pixel space. The processed datamaps will be saved in the `results/datamaps_processed/<NAMEOFFOLDER>` directory.