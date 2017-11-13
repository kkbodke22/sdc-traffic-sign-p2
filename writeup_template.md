# Traffic Sign Classifier [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Goals of the project
* Load the data set
* Analyze the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Summarize the results with a written report

### Jupyter Notebook

* Source Code: [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb)

### Basic summary of the data set

* Download dataset: [traffic-signs-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory visualization on the data set

The bar chart shows the data distribution of the training data. Each bar represents one class (traffic sign) and how many samples are in the class. The mapping of traffic sign names to class id can be found here: [signnames.csv](./signnames.csv)


### Design and Test a Model Architecture

#### Preprocessing

The images of the training dataset have 3 color channels. I reduced the channels to only one. I converted the images to YCrCb color space and use the y-channel. Here you can see how it looks like:

|original image|preprocessed image
|----|----|
|![original image](./images/original_image.png "original image")|![preprocessed  image](./images/preprocessed_image.png "preprocessed image")|

The main reason why I reduced to one channel is because of reducing the amount of input data, training the model is significantly faster. The color of traffic signs should not be importent for classification. They are designed that even color blind people can identify them.

I normalized the data before training for mathematical reasons. Normalized data can make the training faster and reduce the chance of getting stuck in local optima.

#### Model Architecture
 
 I use a convolutional neuronal network to classify the traffic signs. The input of the network is an 32x32x1 image and the output is the probabilty of each of the 43 possible traffic signs.
 
 My final model consisted of the following layers:

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x48|
| Max pooling			| 2x2 stride, 2x2 window						|28x28x48|14x14x48|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x48|10x10x96|
| Max pooling			| 2x2 stride, 2x2 window	   					|10x10x96|5x5x96|
| Convolution 3x3 		| 1x1 stride, valid padding, RELU activation    |5x5x96|3x3x172|
| Max pooling			| 1x1 stride, 2x2 window        				|3x3x172|2x2x172|
| Flatten				| 3 dimensions -> 1 dimension					|2x2x172| 688|
| Fully Connected | connect every neuron from layer above			|688|84|
| Fully Connected | output = number of traffic signs in data set	|84|**43**|

#### Model Training

I trained the model on my local machine with a GPU (GeForce GTX 1060 6GB). Compared to my cpu, the training was much faster.


Here are my final training parameters:
* EPOCHS = 35
* BATCH_SIZE = 128
* SIGMA = 0.1
* OPIMIZER: AdamOptimizer (learning rate = 0.001)

My results after training the model:
* Test Accuracy = **93.9%**

#### Solution Approach

My first implementation was LeNet-5 shown in the udacity classroom. I modified it to work with the input shape of 32x32x3. It was a good starting point and I get a validation accuracy of about 90%, but the test accuracy was much lower (about 81%). So I modified the network and added more convolutional layer, did some preprocessing and looked which changes give better results. You can see my final model architecture above. I experiment with dropout as well, but I couldn't see any improvements. Training for more than 35 epochs do not increase the validation accuracy. I trained the network for 50 and more epochs, but I get a slightly decreasing accuracy. So I decided to stop training after 35 epochs.
