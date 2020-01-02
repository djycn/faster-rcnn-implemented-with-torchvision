
# *Faster* R-CNN: implemented with torchvision

### arguments:
 - --backbone_name -> the backbone you want to use, support `vgg16`, `resnet34`, `resnet50`.
 - --backbone_path -> backbone weights' path. 
 - --data_path -> data path to load training data.
 - --resume -> checkpoints to resume from.
### examples:  

	   - train: python train.py --backbone_name=resnet50 --backbone_path=models/resnet50-19c8e357.pth --data_path=data/custom

	   	    python train.py --backbone_name=resnet34 --backbone_path=models/resnet34-333f7ec4.pth --data_path=data/custom

		    python train.py --backbone_name=vgg16 --backbone_path=models/vgg16-397923af.pth --data_path=data/custom

		    python train.py --backbone_name=resnet50 --backbone_path=models/resnet50-19c8e357.pth --resume=checkpoints/resnet50/faster_rcnn_model_2.pth
                  
	   - detect: python detect.py

		     python detect.py --resume_model=checkpoints/resnet50/faster_rcnn_model_2.pth

### data formats

In the `$data_path_dir`, all images for training and evaluating should be stored in `.jpg` format and in folder `images/`. Iits labels should be in the form of `class_index x1 y1 w, h` (x1, y1, w, h all range from 0 to 1, where x1 and y1 stand for the centroid coordinates for the bbox while w and h stand for the width and height.)for all bounding boxes, and the `class index` shall start from 1.

All classes' names shall be stored in `classes.names` one line for each class name with `background` as the default class name. The training images' directory shall be stored in `train.txt` like `0001.jpg`, and the validating images shall be stored in `valid.txt` resembly.
	    

### License

Faster R-CNN implenmwnted with torchvision is released under the MIT License (refer to the LICENSE file for details).



