# Real-ESRGAN-bicubic

## Absutract <br>
&emsp; You can use this [Real-ESRGAN-bicubic] to train and test yourself.
In this repository, for simplicity, we will only perform super-resolution on images that have been compressed using the bicubic method.
There are two types of test codes, "esrgan_bicubic.py" and "esrgan_cross.py", and it is generally better to use "esrgan_bicubic.py".
The "esrgan_cross.py" uses a special loss function.<br>
&emsp; At the top of each training code, a section called "variable" is provided.
You can change the learning rate, the number of epochs, and the depth of the layers of the generator.
The input image size is defined, and the default value is 1024. 
If you want to change the size, you can change "crop_pad_size = 1024" in line 58 of "esrgan_data.py".
The other default settings are as follows.　<br>

&emsp;### default settings <br>


&emsp;### esrgan_bicubic.py <br>

&emsp;### esrgan_cross.py <by>
  
