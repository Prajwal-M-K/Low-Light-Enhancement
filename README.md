This repository contains models and scripts for detecting & enhancing low-light videos, as part of a summer internship project at CIE, PES University. The project explores both traditional and deep learning–based techniques, focusing on improving visibility in street-level low-light scenes.

Two Enhancement models were explored:
  
  1.**CLAHE** - Contrast Limited Adaptive Histogram Equalization, run using the Run_CLAHE.py script. This is was a model directly taken from the OpenCV library.
  
  2.**UNet** - Uses the UNet architecture, and is trained on the LoLI-Street dataset (https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street)
  
The detetction model was built using the pretrained ConvNeXt architecture, and can be run using the Classify.py file.

All the training scripts were run on Kaggle and are hence coded specifically to Kaggle. The scripts to run the models were created as a part of our backend, and can be run using the following command (UNet as reference) : 

python Run_UNet.py --input path/to/video.mp4 --output output.mp4
