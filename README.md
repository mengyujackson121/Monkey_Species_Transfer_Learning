# Monkey Species Identification

## Business Problem: 
Conservation groups in wildlife protection areas have limited anti-poaching resources and need help focusing them on the most effective areas. They have motion-triggered cameras set up throughout the conservation zone, which capture too many photos each day to be reviewed by hand. Not all species are targeted by poachers, and not all species are equally at risk. They need an automated process to identify whether a monkey is in each photo, and to identify the species of each monkey identified. They can combine this data with location and timing information from each photo to decide how to use anti-poaching resources most effectively.


## Technical Issues + Constraints
We have only a small dataset for training and validation (~150 images per class), and the budget for this project is limited. This constrains machine size, training time, and ability to obtain additional human labeled data. 

## Strategy
To solve the business problem under these constraints, we will use Transfer Learning to adapt pre-trained models which perform very well on the ImageNet task (https://www.image-net.org/challenges/LSVRC/) to the Monkey Species classification problem.


## Prerequisites: Download and Merge Datasets


### Setup
Create a directory to contain the images. The remainder of the "Prerequisite" steps should be completed from within this directory.

### Monkey Species Images
Most data comes from https://www.kaggle.com/slothkong/10-monkey-species. These 10 monkey species are the main classification targets. Download and extract the `training` and `validation` folders. Rename `training` to `train` and `validation` to `val`.

### Generic Wildlife Images
The rest of the data comes from https://www.kaggle.com/virtualdvid/oregon-wildlife. All of these images will be given the same label `nm` (for "**N**ot a **M**onkey"), because the wildlife cameras will be capturing photos not involving monkeys as well. Download and extract the full `oregon-wildlife` folder into the image directory.

### Merge

Create subdirectories `train/nm` and `val/nm`.

Run the script `dataclean.py` to move images from the `oregon_wildlife/<class-name>` directory into the `training/nm` and `val/nm` directories. We use a small number of images to keep class sizes balanced.

If you need to re-run this script, remove daty from the `nm` folders first because the script will add new images without replacing old images (resulting in imbalanced classes). 

The training/analysis pipelines are not designed with imbalanced classes in mind.


## Step 1: Import Libraries and Data, Set Variables

## Step 2: Train Pytorch Models


### Reference
The code in this section (and `pytorch_transfer_learn.py`) is adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#helper-functions with the following changes:
* Multi-class output instead of binary output (which means changing loss fn, accuracy metric, etc.)
* Trains 6 models instead of a single model
* Saves models after training (and loads instead of retraining if `retrain == False`
* Pytorch Ignite is used for visualizing results using a confusion matrix


### Analysis Note
This training is done as a preliminary investigation in order to decide whether to invest more time in improving a Pytorch or Keras based approach for the final model. Some typical model evaluation metrics and training graphs are left out because they are unnecessary for this decision.


## Pytorch Results
DenseNet clearly outperforms all other Pytorch models, incorrectly classifying only 1 image in the validation set. The model proves robust in the face of monkeys facing away from the camera, multiple monkeys, poor lighting, monkeys facing away from the camera, and monkeys of different ages (juveniles often look different from adults).


## Keras Training
### Reference
The code in this section (and `keras_transfer_learn.py`) is adapted from https://keras.io/guides/transfer_learning/ with the following changes:
* Multi-class output instead of binary output (which means changing loss fn, accuracy metric, etc.)
* 300x300 img size (similar to largest Pytorch Model) rather than 150x150
* Saves models after training (and loads instead of retraining if `retrain == False`
* Fixed Training and Validation directories are used to match Pytorch setup


## Keras Results
The keras model performs equally to the best Pytorch Model, incorrectly classifying only 1 image.  The model proves robust in the face of monkeys facing away from the camera, multiple monkeys, poor lighting, monkeys facing away from the camera, and monkeys of different ages (juveniles often look different from adults).

The single mis-classified image is an individual with a unusual coat color for his species (white instead of brown), incorrectly classified as a species which has a typically white coat color. The second highest class probability for this image is correct, showing the model does see traits associated with both species in this image.



# Overall Results + Recommendation

## Overall Result
The current accuracy of the Keras model or the Pytorch is sufficient for our client. The model proves robust in the face of monkeys facing away from the camera, multiple monkeys, poor lighting, monkeys facing away from the camera, and monkeys of different ages (juveniles often look different from adults).

Because they get different samples wrong, and each has an extremely high degree of validation accuracy, it is difficult at this time to select a single model or measure improvement from additional training.


## Recommendation
We recommend running both the Pytorch and Keras Models for inference, and using human review only when they disagree. If models are found to be wrong (or disagree) often, we recommend investing in further model development at that time (likely just additional training data).
