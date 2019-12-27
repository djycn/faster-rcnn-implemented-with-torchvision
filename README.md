
# *Faster* R-CNN: implemented with torchvision

commands:
 - --backbone_name -> the backbone you want to use, support vgg16, resnet34, resnet50
 - --backbone_path -> backbone weights' path 
 - --data_path -> data path to load training data
examples:  train: python train.py --backbone_name=resnet50 --backbone_path=models/resnet50-19c8e357.pth --data_path=data/custom
	   	  python train.py --backbone_name=resnet34 --backbone_path=models/resnet34-333f7ec4.pth --data_path=data/custom
		  python train.py --backbone_name=vgg16 --backbone_path=models/vgg16-397923af.pth --data_path=data/custom
	   detect: python detect.py
### data formats

all images for training should be stored in .jpg format and in folder images/, all labels should be in the form of class_index x1 y1 x2 y2 for all bounding boxes, and the class index shall start from 1
all classes with background as the default class; names for all classes shall be stored in classes.names. The training image;s directory shalled be stored in train.txt like 0001.jpg, and the validating images
shall be stored in valid.txt resembly.
	    

### License

Faster R-CNN implenmwnted with torchvision is released under the MIT License (refer to the LICENSE file for details).


