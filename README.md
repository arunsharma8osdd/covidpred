# CovidPred
Classification of chest X-ray images into COVID-19 and Major Infectious Diseases

This repository contains:

**1. Sample training set images:** train.zip

**2. Sample validation set images:** test.zip

**3. Scripts used to augment the original images:** CLoDSA_augmentation_scripts.zip

**4. Five types of CovidPred prediction models:** train_rotate_120_angle_24_epochs_model.py, train_rotate_140_angle_24_epochs_model.py, train_combined_24_epochs_models.py, train_combined_49_epochs_models.py and train_combined_101_epochs_models.py

**5. Code used for training the rotated 120 degree images (augmentation) based CovidPred prediction model (24 epochs based):** train_rotate_120_angle_24_epochs_model.py

**6. Code used for training the rotated 140 degree images (augmentation) based CovidPred prediction model (24 epochs based):** train_rotate_140_angle_24_epochs_model.py

**7. Code used for training the original images and multiple augmentation based CovidPred prediction model (24 epochs based):** train_combined_24_epochs_models.py

**8. Code used for training the original images and multiple augmentation based CovidPred prediction model (49 epochs based):** train_combined_49_epochs_models.py

**9. Code used for training the original images and multiple augmentation based CovidPred prediction model (101 epochs based):** train_combined_101_epochs_models.py

**10. Code used for testing or external validation of the CovidPred prediction model (rotated 120 degree images; 24 epochs based):** validate_rotate_120_angle_24_epochs_model.py

**11. Code used for testing or external validation of the CovidPred prediction model (rotated 140 degree images; 24 epochs based):** validate_rotate_140_angle_24_epochs_model.py

**12. Code used for testing or external validation of the CovidPred prediction model (Combined Model 1; 24 epochs based):** validate_combined_24_epochs_model.py

**13. Code used for testing or external validation of the CovidPred prediction model (Combined Model 2; 49 epochs based):** validate_combined_49_epochs_model.py

**14. Code used for testing or external validation of the CovidPred prediction model (Combined Model 3; 101 epochs based):** validate_combined_101_epochs_model.py

**15. Assisting Python code (used in models training Python scripts for datasets preparation):** dataset.py


# Frequently asked questions:

**Note:** The training and validation set images must be supplied in unzipped format only. 

**Question. Can I download the already trained models used in present study?**

**Answer.** Yes, you can download all the five trained models from our website at: http://14.139.62.220/covid_19_models/

**Question. What are the dependencies or library and operating system (OS) requirements to run these codes?**

**Answer.** The library requirements are as follows:

**OS:** Linux / Windows / Mac OS (Any OS with Python 2.7.15+ and Python modules such as OS, glob, shuffle, numpy, time, sys and matplotlib modules installed on it)

**Other Libraries required:** Open CV2, tensorflow-1.5.0


**Question. How to train deep learning models using above given codes?**

**Answer.** The users have to store their training set images in the training dataset directory (as specified in python scripts) as given in "train.zip" and run command as given below:

**I) To develop 120 degree rotated images (24 epochs based) model:**

python train_rotate_120_angle_24_epochs_model.py

**II) To develop 140 degree rotated images (24 epochs based) model:**

python train_rotate_140_angle_24_epochs_model.py

**III) To develop Combined Model 1 (24 epochs based):**

python train_combined_24_epochs_models.py

**IV) To develop Combined Model 2 (49 epochs based):**

python train_combined_49_epochs_models.py

**V) To develop Combined Model 3 (101 epochs based):**

python train_combined_101_epochs_models.py

The above mentioned codes have all steps in common except the number of iterations, filters size, filters number, titles of training plots, X-axis values range, trained models names and model's performance showing plots names. These will train deep learning models and save. For each epoch, 90% data will be used in training the models and rest 10% for internal validation of trained models. Moreover, training and validation accuracy along with validation loss will be printed on terminal window. After last epoch, model will save automatically (with model's name given within the script) and a plot/figure will also generate. The generated figure will have training and validation accuracy along with validation loss plotted on y-axis while number of epochs on x-axis. 

**Question. How to validate trained models on external validation dataset?**

**Answer.** The user has to supply the validation set images as provided in "validation_set.zip" directory (the names of these validation dataset images directories should be as given in validation python scripts) and run the following command:

**I) To validate 120 degree rotated images (24 epochs based) model:**

python validate_rotate_120_angle_24_epochs_model.py

**II) To validate 140 degree rotated images (24 epochs based) model:**

python validate_rotate_140_angle_24_epochs_model.py

**III) To validate Combined Model 1 (24 epochs based):**

python validate_combined_24_epochs_model.py

**IV) To validate Combined Model 2 (49 epochs based):**

python validate_combined_49_epochs_model.py

**V) To validate Combined Model 3 (101 epochs based):**

python validate_combined_101_epochs_model.py

The above mentioned codes have all steps in common except that the respective prediction models to be used for validation. The output of codes will be variety-wise prediction accuracy (%) on the terminal window.


**Question. How to augment the original images?**

**Answer.** The user may use the json scripts given within "CLoDSA_augmentation_scripts.zip" for the augmentation of their original images. The input images in the present case are given in directory named "train" and the output will be stored within the new directories (to be created by users) after running a particular augmentation script e.g.,

clodsa aug_CLoDSA_rotate_45.json

The augmented images will be stored in newly created directory. These may be further used for training and testing of prediction models. 


**Note:** Originally, the deep learning models training and validation Python scripts were downloaded from https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier and customized according to requirements of present study.


