# Real-ESRGAN-bicubic

## Absutract <br>
&emsp; You can use this [Real-ESRGAN-bicubic] to train and test yourself.
In this repository, for simplicity, we will only perform super-resolution on images that have been compressed using the bicubic method.
However, the structure of the generator and discriminator is exactly the same as that of the official Real-ESRGAN.
There are two types of training codes, "esrgan_bicubic.py" and "esrgan_cross.py", and it is generally better to use "esrgan_bicubic.py".
The "esrgan_cross.py" uses a special loss function.<br>
&emsp; At the top of each training code, a section called "variable" is provided.
You can change the learning rate, the number of epochs, and the depth of the layers of the generator.
The input image size is defined, and the default value is 1024. 
If you want to change the size, you can change "crop_pad_size = 1024" in line 58 of "esrgan_data.py".
The other default settings are as follows.　<br>

### default settings <br>
- learning rate : 0.0001
- epochs : 10
- image size : 1024x1024

### training <br>
- esrgan_bicubic.py <br>
- esrgan_cross.py <br>

### test <by>
- esrgan_test.py <br>
  
## Dataset Preparation <br>
dataset : https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset?select=train
- recommendation <br>
&emsp; image-size : 1024x1024 <br>
&emsp; number of images : 1000 <br>
