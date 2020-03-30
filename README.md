# CovidPred
Classification of chest X-ray images into COVID-19 and Major Infectious Diseases
This repository contains:

**1. Sample training set images:** train.zip

**2. Sample validation set images:** test.zip

**3. Scripts used to augment the original images:** CLoDSA_augmentation_scripts.zip

**4. Three CovidPred prediction models:** 

**5. Code used for training the CovidPred prediction model (24 epochs based):** 

**6. Code used for training the CovidPred prediction model (49 epochs based):** 

**7. Code used for training the CovidPred prediction model (101 epochs based):** 

**8. Code used for validation of the CovidPred prediction model (24 epochs based):** 

**9. Code used for validation of the CovidPred prediction model (49 epochs based):** 

**10. Code used for validation of the CovidPred prediction model (101 epochs based):** 

**11. Assisting Python code (used in models training Python scripts for datasets preparation):** dataset.py


# Frequently asked questions:

**Note:** The training and validation set images must be supplied in unzipped format only. 

**Question. What are the dependencies or library and operating system (OS) requirements to run these codes?**

**Answer.** The library requirements are as follows:

**OS:** Linux / Windows / Mac OS (Any OS with Python 2.7.15+ and Python modules such as OS, glob, shuffle, numpy, time, sys and matplotlib modules installed on it)

**Other Libraries required:** Open CV2, tensorflow-1.5.0


**Question. How to train deep learning models using above given codes?**

**Answer.** The users have to store your training set images in directory named "train" as given in "train.zip" and run command as given below:

**I) To develop 24 epochs based model:**

python train_model_24_epochs.py

**II) To develop 49 epochs based model:**

python train_model_49_epochs.py

**II) To develop 101 epochs based model:**

python train_model_101_epochs.py

The above mentioned codes have all steps in common except the number of iterations and titles of training plots. These will train deep learning models and save. For each epoch, 90% data will be used in training the models and rest 10% for internal validation of trained models. Moreover, training and validation accuracy along with validation loss will be printed on terminal window. After last epoch, model will save automatically (with model's name given within the script) and a figure will also generate. The generated figure will have training and validation accuracy along with validation loss plotted on y-axis while number of epochs on x-axis. 

**Question. How to validate trained models on external validation dataset?**

**Answer.** The user has to supply the validation set images as provided in "validation_set.zip" directory and run the following command:

**I) To validate 24 epochs based model:**

python validate_model_24_epochs.py

**II) To validate 49 epochs based model:**

python validate_model_49_epochs.py

**II) To validate 101 epochs based model:**

python validate_model_101_epochs.py

The above mentioned codes have all steps in common except that the respective prediction models to be used for validation. The output of codes will be variety-wise prediction accuracy (%) on the terminal window.


**Question. How to augment the original images?**

**Answer.** The user may use the json scripts given within "CLoDSA_augmentation_scripts.zip" for the augmentation of their original images. The input images in the present case are given in directory named "train" and the output will be stored within the new directories (to be created by users) after running a particular augmentation script e.g.,

clodsa aug_CLoDSA_rotate_45.json

The augmented images will be stored in newly created directory. These may be further used for training and testing of prediction models. 


**Note:** Originally, the deep learning models training and validation Python scripts were downloaded from https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier and customized according to requirements of present study.


