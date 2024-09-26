# Multi-Lesion Lung SABR Dose Prediction
## Overview
This repository contains code and instructions for training model in the paper: [Predicting the 3-Dimensional Dose Distribution of Multilesion Lung Stereotactic Ablative Radiation Therapy With Generative Adversarial Networks](https://www.redjournal.org/article/S0360-3016(24)03175-4/abstract)

The files located in the parent directory are used to train the model. The files located in the [SlicerJupyterCode](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/tree/main/SlicerJupyterCode) folder are used for preprocessing (converting DICOM files to numpy files). However, any preprocessing method should work. The files located in the [SlicerExtension](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/tree/main/SlicerExtension) folder are used to allow for visualizing dose predictions in 3DSlicer.

Before you get started, update the [config.yml](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/blob/main/config.yml) file approrpriately. 

## Table of Contents
- [Preprocessing Data](#preprocessing-data)
- [Model Training](#model-training)
- [Visualization](#visualization)
- [Citation](#citation)

## Preprocessing Data
Note: This step is optional if you have another method of preprocessing your data.

We use 3DSlicer and [rt-utils](https://github.com/qurit/rt-utils) to convert the DICOM data (planning CT, RTStruct, RTDose) into numpy files which are used to train the GAN. We do this using the SlicerJupyter extension. Instructions on setting up SlicerJupyter can be found [here](https://github.com/Slicer/SlicerJupyter) and [here](https://github.com/Marcus-Milantoni/Slicer_scripting_tutorial). You will also have to install the [SlicerRT](https://slicerrt.github.io/Download.html) extension. 

Also, you will need a .csv file that contains the required information about each patient. This is the DATA_FILE in config.yml. See [SampleDataFile.csv](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/blob/main/SampleDataFile.csv) for an example.

First, use [VoxelizeStructures](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/blob/main/SlicerJupyterCode/VoxelizeStructures.ipynb) to convert the RTStruct to numpy files. Then, use [ConvertDICOMtoSlicerMRB](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/blob/main/SlicerJupyterCode/ConvertDICOMToSlicerMRB.ipynb) to create Slicer MRB files. Finally, use [ConvertSlicerMRBToNumpy](https://github.com/edwardwang1/MultiLesionLungSABRDosePrediction/blob/main/SlicerJupyterCode/ConvertSlicerMRBToNumpy.ipynb) to generate the numpy files.

## Model Training
We use Python 3.9 and PyTorch 2.0.1. We also use tensorboard for logging, but this is not necessary if the the logging parts are removed.
Make sure that the NUMPY_SAVE_PATH folder contains all the .npy files of your patients. Also ensure that TRAIN_PATIENT_LIST and TEST_PATIENT_LIST contain the IDs of your patients in the training and testing set respectively. 


Once the data is there, simply run

```
python train.py
```

## Visualization
We created an extension for 3DSlicer that allows you to run the model within Slicer. Make sure to have [SlicerRT](https://slicerrt.github.io/Download.html) installed, as well as [SlicerPyTorch](https://github.com/fepegar/SlicerPyTorch). To load the module into Slicer, go into "Application Settings" -> "Modules" -> "Additional module paths" and add the SlicerExtension folder.

![image](https://github.com/user-attachments/assets/81d670c7-9a31-4469-ae27-7602b6de9edd)


## Citation
If you use this work, please cite:

Predicting the 3-Dimensional Dose Distribution of Multilesion Lung Stereotactic Ablative Radiation Therapy With Generative Adversarial Networks.
Wang, Edward et al. International Journal of Radiation Oncology, Biology, Physics, Volume TBD, Issue TBD



