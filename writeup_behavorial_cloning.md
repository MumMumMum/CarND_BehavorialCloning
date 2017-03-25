
# The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

# My project includes the following files:
* Model_Generator.ipnb ,(I used notebook so that i could plot and visualize the data and network)containing the script to create and train the model, It all has pipe line for reading images through genrator , prepocessing and agumenation
* drive.py for driving the car in autonomous mode, and along with some crop and resizing of image as we had resized the orignal image for our model.
* model_20k.h5 containing weights of trained convolution neural network 
* writeup_report as md with video Link here: https://www.youtube.com/watch?v=sfssyduAbag&feature=em-upload_owner


# Code:
Model_Generator.ipnb contains :
1)Code for reading the driving_log_3.csv,Thos file has 9000 records .Added some HSV tranformation images and Images at turn after bridge where the model was failing initally.

2)The code for reading the entries from CSV, reading images turning it into arrays for a batch of 32 using generator

3)The generator does some preprocessing and agumentation.

4)Genertor feeds the  model and saving the weights of neural network.

5)The model output is plotted for training vs validation loss.

while there were lot of other files used to write for data balancing exploring.


# Inital Methods :

Data given in driving_log.csv had 9000 entries out of which 4000 were zero angle.
As I wanted to balance  data I worte a script to read
image and add new agumented image if sterring angle between 0.3 to 0.6 .I wanted to suppy this array to genertor .
I had nod idea of how generator could  balance images.
But I failed to run model due to heavy image array already generated.
So finally i discarded this approach, i balanced the csv by removing 4000 odd zero angle images and then fed this new 
driving_log5000.csv to model.ipnb I achieved good performance on my CPU with this. But the car was wobbly 



# Idea that worked: 
    
Finaly as my mentor suggested dont remove any data(zero angle data records.), I stuck to that idea and kept adding images until my data got balanced.

# Data collection:

I used Udacity track1 data .
Then there were so many methods used :
1) I deleted few records of 0 angle and increased the large angle data amd was trying to run the model

2) Just Udacity Data split in 80 20 ratio

3) Trying to balance data for all angle type in generator code for each batch(This is not yet thru)

4) adding 1000 more images of large angle >.3 in HSV mode to data

5)Added data for a specific turn






# Data Preprocessing and agumentation
Image inital size was 160,320,3 So generate takes in images of 32 per batch and for preprocessing.First cropped the image along Y axis 40 from top 20 from bottom and then along x axis 10 from left and ten from rt.Got down image to 100 * 300 and then resized to 64* 64
<img src="output_images\imageresize.png">
This helps in reducing memory requirements and remove redundant pixels which helps in better approximation
 
Then Further it reads images from left and right camera to which the measurement of anglewe add offset of 0.25 and -0.25 to keep the car in center . 

Further preprocseeing is done in model using lambda layer which normalizes in range of -1 to 1,This helps 
the optimizer to optimize the model faster.  I removed this as i saw better performance without lambda lader but manually centring data in genertor.



Data Agumentation flip image horizontaly, crop image, add random brightness, add random shadow was used.Added HSV and YUV images for large angles.

<img src="output_images\display.png">



```python

```

# The inital data distribution:

<img src="output_images\image_original.png">

# The data distribution after adding HSV images and images for a specific distribution:

<img src="output_images\image_for_95353.png">

# Model Architecture :


This is model is similar to NIVIDIA network and is little altered to tune to this project.


Model ======>24@5*5 filter===>36 @ 5*5 =====> Pooling 2*2=====>dropout 0.2====>48 @ 3*3=====>Pooling 2*2 dropout 0.4==========>64 @ 3*3 ====>dropout 0.4======>Flatten====>FC 1024=======>dropout 0.2=====>FC 512======>drop put 0.2=========>FC10====>FC10====>Classifier OUTPUT.

Here we have used 2 layers of 5 by 5 filters, 2 filters of 3*3 , Droput layers, Fully connected layers, MAx Poolong layers, relu activation.
The previous reviewer had suggested to use batch normalization which worked wonders from me.
But the lesson Suggested crop layer and Lambda normalization were of no use for me.
Lambda normalization I found compuation swaying the car. And crop layer  taxed my CPU.
The model was significantly different from all the previous lab problems as we did first time a non classification output which made this pb of regression.



The prev model had preprocessing Layers like crop and Lambda(for normalization).
But now I have batchNormalization instead.
I tried to use keras ImageGenerator flow and failed,I have to yet try flowfromDirectory
I want to use Image Generator as My agumentaing code I need to still tune with.
The reason I was not sure of Keras Image genertaor was I was not
sure of how will it change Angle meauremnt with image change
The output from each layer is as Follows

# 1op shape after conv 24 of 5x5 (None, 30, 30, 24)
# 1op shape after conv 36 of 5 x5 (None, 26, 26, 36)
# 3op shape after conv 48 of 3 x3 (None, 6, 6, 48)
# 4op shape after conv 16 of 3 x3 (None, 4, 4, 64)
# flatten (None, 1024)

I am pretty Ok with the model.


The model was trained for 9 epoch and for 30000 images through Generator.Even tough it work for 5 epoch I have tested till 15 epoch the model never shows sign of overfitting.There is a stady loss and constant value reached.
Data agumentaion was used , camera left and right images were used, HSV,YUV images were used, Random shadow, Flipp of images as well were used to train model.
Validation and training Loss over several epoch Looks like below: There is a very nice drop in both training and validation loss and is staedy.That means there is no overfittong of model.

# My Reflections:
A very good project and opportunity Udacity gave me in this project. This project involed recollection of all topic we studied inside the course.The best part of this project is  fails and re fails and re fails unti you touch all the hyper parameters of a learning model.Thank you for this excellent opportunity.
The Training pgm trained us on classification and validated us on regression :) sticking to philosophy of testing on unseen data. Bravo!!!
Also I wanted to generate a generator in which data in each batch is balanced, I tried it but didnt get through.If I would have done that I think my model would have run in 15k images.
Had to discard lot of models and reuses old models as well.Had to delete data add data.
Had to add noise brigthness and preprocess data.
I am very happy to have done this ... but still i feel there are untouched corners like i am not able to get back and improvize my work as i am just able to finish the assigned work in given time.

The most reflecting point is when I add images for a specific turn , That mean in real term I am overfitting a model by giving him more info for a specific turn where he is failing to predict right angles.what do you think about the same...??








