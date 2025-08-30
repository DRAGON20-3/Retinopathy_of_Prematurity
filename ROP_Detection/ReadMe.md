The implementations of ROP Detection provided here. This implementation consists of a classification method to detection the normal or disease eyes conditions on the taken images of the vessels of the eyes from prematures. The pipeline of this project is implemented in `Train.py` which steps below are involved subsequently:

1. Importing packages: Basic and common packages are used. Almost all of the codes and processes are developed manually to clarify details and bring more control on the codes. Most of the developments are done by the numpy, tensorflow, PIL (aka pillow), and sklearn (aka Scikit-Learn) packages.

2. Setting the global parameters (such as number of epochs, batch size, etc.).

3. Loading datasets: using the directory next to the code. Some samples of images are provided as a reference of how to set and located the data and how the data look like.

4. Pre-Processing inputs and labels to prepare them for feeding to the model.

5. Model development: The model is created from scratch whose details are available which make it more flexible for similar tasks. The model is developed using the tensorflow package.

6. Training and Validation Loop: A loop which is specified to handle the load of the data is developed. Similar to online learning, in each iteration of this loop, a set of samples will be selected randomly and fairly with the same number of each class. Next, they will undergo some preprocessing (as mentioned in step 4) for both inputs and labels. Afterward, the augmentation will increase the number of samples if activated. Finally the model will use them for training for specific number or epochs per iteration and report the training and validatino metrics. The history of metrics in this loop and the model will be stored in a determined address.

7. Evaluation metrics reporting: The classification performance is reported for more evalution. It includes Precision, Recall, and F1-Score for each class using the sklearn package.

Some comments are added for making the codes more legible and clear.

<!--This model will be trained and test on images of retina for Retinopathy of Prematurity Detection in infants.-->

