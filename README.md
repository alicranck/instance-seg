# instance-seg
Instance segmentation with deep metric learning and context.

This work follows https://arxiv.org/abs/1708.02551 and https://arxiv.org/abs/1703.10277

It is a PyTorch implementation of an instace-segmentation model that embeds the pixels in d-dimensional
embedding space, and uses clustering to find the instances.

In this work I have added context - an RNN that runs over the embeddings and outputs a context vector
per pixel. This significantly improves the results of the model.

## Implementation details
This model uses data in the format of the PASCAL VOC dataset, i.e. JPEG images with PNG labels containing 
ground truth instace segmentations. To use the model, download the code, **change all directory paths 
at config.py and train.py, and set the hyperparameters at config.py as you wish.**
run *python train.py*. The model will be saved after each epoch to the checkpoints directory you
defined. Please note the required packages at the requirements file.

The model does not require a GPU, but to be fair it is nearly impossible to train without one as the model is 
fairly large. 

If you wish to use COCO data, please see https://github.com/alicranck/coco2voc or https://github.com/cocodataset/cocoapi to 
convert the COCO annotations to VOC style segmentations.

Some warnings are shown if you use PyTorch version 0.4.0 and up. I chose to maintain some backward compatability
and not remove these warnings. It should be fairly simple to do if you do run PyTorch 0.4.0.

## Disclaimer
I have not tested this code extensively as it was used for a personal research project. Please let me know if you find any 
errors inherent in the code and I will try to adress these as necessary.


Some results obtained after training the network on COCO for ~15 epochs with context:
![sample_image_1](/images/sample_1.png)
![sample_image_2](/images/sample_2.png)
![sample_image_3](/images/sample_3.png)

without context:


![sample_image_no_1](/images/sample_no_1.png)
![sample_image_no_2](/images/sample_no_2.png)
![sample_image_no_3](/images/sample_no_3.png)


