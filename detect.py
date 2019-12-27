import argparse
from datasets import *
from PIL import Image, ImageDraw,ImageFont
from model import *
import cv2 as cv
import random

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Training')
    parser.add_argument('--backbone_name', default='resnet50', help='which backbone to use')
    parser.add_argument('--data_path', default='data/custom', help='dataset path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                        help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                        help='beta2 for adam')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--eval_interval', default=1, type=int, help='the evaluating interval')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--iou_thres', default=0.1, type=float, help='iou threshold for evaluating')
    parser.add_argument('--score_thres', default=0.8, type=float, help='score threshold for inferencing')
    parser.add_argument('--resume_model', default='checkpoints/faster_rcnn_model_3.pth', help='resume from checkpoint')
    parser.add_argument('--test_only', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--detect_path', default='output', help='path to save the detecting results')
    parser.add_argument('--output_dir', default='checkpoints/', help='path where to save')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--parallel', default=False, help='if distribute or not')

    args = parser.parse_args()

    return args


args = get_args()
current_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir, args.data_path)
train_path = os.path.join(data_path, 'train.txt')
test_path = os.path.join(data_path, 'valid.txt')

names_path = os.path.join(data_path, 'classes.names')
dataset = ListDataset(train_path)
with open(names_path, "r") as file:
    classes = file.readlines()
classes = [cls.strip() for cls in classes]
num_classes = len(classes)
print('num_classes', num_classes)
with open(test_path, "r") as test_file:
    test_files = test_file.readlines()
test_files = [item.strip() for item in test_files]

vgg16_path = 'models/vgg16-397923af.pth'
resnet_path = 'models/resnet50-19c8e357.pth'
# device = torch.device(args.device)
device = torch.device('cuda')

# faster_rcnn = vgg_model(num_classes, vgg16_path)
faster_rcnn = fasterrcnn_resnet_fpn(args.backbone_name, resnet_path, pretrained=False, num_classes=num_classes)
faster_rcnn.to(device)
if args.resume_model:
    print('loading saved model')
    checkpoint = torch.load(args.resume_model, map_location='cpu')
    faster_rcnn.load_state_dict(checkpoint['model'])
    

    
    
def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    
    return (b, g, r)


def detect(model, images, device, score_thres, classes):
    model.eval()
    detect_imgs = []
    for image in images:
        img_tensor = torch.from_numpy(image / 255.).permute(2, 0, 1).float().cuda()
        detect_imgs.append(img_tensor)
    
    # for j, img in enumerate(images):
    #     test_img = transforms.ToPILImage()(img)
    #     test_img.save('test{}.jpg'.format(j))
    detect_imgs = [detect_img.to(device) for detect_img in detect_imgs]
    # print(images[0].shape)

    predictions = model(detect_imgs)
    result_imgs = []
    for img_i, (image, prediction) in enumerate(zip(images, predictions)):
        # print(prediction)
        boxes = prediction['boxes'].detach().cpu().data.numpy()
        # print(boxes)
        labels = prediction['labels'].detach().cpu().data.numpy()
        scores = prediction['scores'].detach().cpu().data.numpy()
        if len(boxes):
            for idx in range(boxes.shape[0]):
                if scores[idx] >= score_thres:
                    x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                    name = classes[int(labels[idx])]
                    cv.rectangle(image, (x1, y1), (x2, y2), random_color(), thickness=1)
                    cv.putText(image, text=name, org=(x1, y1 + 10), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.3, thickness=1, lineType=cv.LINE_AA, color=(55, 55, 200))
        result_imgs.append(image)
    return result_imgs
    
    
if not os.path.exists(args.detect_path):
    os.mkdir(args.detect_path)
    
    
def main():
    for j, file in enumerate(test_files):
        src_img = cv.imread(file)
        # img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
        # img = transforms.ToTensor()(src_img)
        # test_img = transforms.ToPILImage()(img)
        # test_img.save('test{}.jpg'.format(j))
        inputs = [src_img]
        image = detect(faster_rcnn, inputs, device, args.score_thres, classes)[0]
        cv.imwrite(args.detect_path + '/{}.jpg'.format(j), image)
        
        
if __name__ == "__main__":
    main()