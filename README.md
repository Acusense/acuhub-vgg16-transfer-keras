## Example: Base VGG16 model w/ transfer learning

* Install and use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

### Instructions

This repository is an example of how to create a Base model you can build, train, and share on [Sensei](https://sensei.com). To get started fork this model to start building your own model.
Below is a description of the file structure, inputs, and outputs of the system so you can access, visualize, train, and share your model on [Sensei](https://sensei.com) 

#### Environment
This entire repository will be built within a scope for your particular model training as defined in Sensei platform. By default your training files will be attached to the 
built Docker at `/training_files`. You can change this setting in the Sensei Platform. 

The following files and folders will be present once your model training has been defined in the Sensei platform: 
``
data.json
config.json
data/
``

For a better understanding of the structure and format of the files check out the [Sensei platform](https://sensei.com)

#### File Structure

* **/scripts**: location where can insert any number of shell scripts to interface with your code. Some example scripts include train.sh, test.sh, etc. 
These scripts will be available via your Sensei page to quickly execute any of the tasks you define, including but not limited to, visualizaiton, training, 
testing, etc. By default train.sh will be your default script, and other scripts can be attached to triggers to execute at particular points during the training or otherwise.

All other files and directories can be defined as you please, the main entrypoints to run your machine learning code will be via your bash scripts. 

#### Outputs
* Standard Out
Console outputs can be monitored on your model training page on the [Sensei platform](https://sensei.com) through the Console window

* Weights 
Weight 

* Statistics and Graphs: 
In our repository training parameters store to a file called `training.csv` within the `training_files/` directory in the following format:
``
(x-axis variable) | (y-axis variable 1) | (y-axis variable 2) | ...
---------------------------------------------------------------------
x1 | y1 | y2
x2 | y1 | y2
``

In this demo repository, the x-axis is epochs and a new row is written for each epoch which includes the following y-axis variables:
training accuracy, validation accuracy, training loss, validation loss. 

* Visualizations:
Visualizations are stored as images in the `visualizations/` folder within the `training_files/` directory. 