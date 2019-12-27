import pdb
import time
import tqdm
from torch import nn
from model import *

import argparse
from datasets import *
import math
import cv2 as cv
from test import evaluate
# load a pre-trained model for classification and return
# only the features
vgg16_path = 'models/vgg16-397923af.pth'
resnet_path = 'models/resnet50-19c8e357.pth'
# device = torch.device(args.device)


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Training')
    parser.add_argument('--backbone_name', default='vgg16', help='which backbone to use')
    parser.add_argument('--backbone_path', default='models/vgg16-397923af.pth', help='backbone model weights path')
    parser.add_argument('--data_path', default='data/custom', help='dataset path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
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
    parser.add_argument('--eval_interval', default=1, type=int, help='the evaluating interval')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--iou_thres', default=0.1, type=float, help='iou threshold for evaluating')
    parser.add_argument('--score_thres', default=0.8, type=float, help='score threshold for inferencing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test_only', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--detect_path', default='output', help='path to save the detecting results')
    parser.add_argument('--output_dir', default='checkpoints/', help='path where to save')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--parallel', default=False, help='if distribute or not')

    args = parser.parse_args()

    return args


def train(model, dataloader, optimizer, lr_scheduler, epoch, device):
    model.train()
    
    # pdb.set_trace()
    loss = 0
    for bi, (images, targets) in enumerate(tqdm.tqdm(dataloader,
                                                     desc="running epoch; {}".format(epoch), total=len(dataloader))):
        # print(images)
        # print(targets)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'image_ids'} for t in targets]
        
        loss_dict = model(images, targets)
        # print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        # print(losses)
        loss_value = losses.item()
        loss += loss_value
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss / len(dataloader)))
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    lr_scheduler.step()
    print("Loss is {} \n".format(loss / len(dataloader)))
    
    
def main():
    args = get_args()
    device = torch.device(args.device)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_dir, args.data_path)
    train_path = os.path.join(data_path, 'train.txt')
    test_path = os.path.join(data_path, 'valid.txt')
    names_path = os.path.join(data_path, 'classes.names')
    
    with open(names_path, "r") as file:
        classes = file.readlines()
    classes = [cls.strip() for cls in classes]
    num_classes = len(classes)
    print('num_classes', num_classes)
    print(args.backbone_name[:3])
    
    if args.backbone_name[:3] == 'vgg':
        faster_rcnn = vgg_model(args.backbone_name, num_classes, args.backbone_path)
    else:
        faster_rcnn = fasterrcnn_resnet_fpn(args.backbone_name, args.backbone_path, pretrained=False, num_classes=num_classes)
    
    faster_rcnn.to(device)
    
    if args.parallel:
        print('Training parallel')
        model = torch.nn.DataParallel(faster_rcnn).cuda()
        
    train_dataset = ListDataset(train_path, augment=True)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    test_dataset = ListDataset(test_path)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )
    params = [p for p in faster_rcnn.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params, lr=0.001,
                                 betas=(args.beta1, args.beta2), eps=1e-08, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    
    if args.resume:
        print('Resume training')
        checkpoint = torch.load(args.resume, map_location='cpu')
        faster_rcnn.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      
        
    for i in range(args.epochs):
        start_time = time.time()
        train(faster_rcnn, trainloader, optimizer, lr_scheduler, i, device)
        # inputs = []
        # for j in range(1, 10):
        #     src_img = Image.open('/home/zhouge/PycharmProjects/Faster-rcnn/data/custom/images/drone_0000000{}.jpg'.format(j))
        #     img = transforms.ToTensor()(src_img)
            # test_img = transforms.ToPILImage()(img)
            # test_img.save('test{}.jpg'.format(j))
            # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
            # inputs.append(img)
        # detect(faster_rcnn, inputs, device, args.score_thres, classes, args.detect_path)
        # pdb.set_trace()
        if i % args.eval_interval == 0:
            evaluate(faster_rcnn, testloader, args.iou_thres, classes, device)
            if args.output_dir:
                save_path = os.path.join(args.output_dir, args.backbone_name)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save({
                    'model': faster_rcnn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args},
                    os.path.join(save_path, 'faster_rcnn_model_{}.pth'.format(i)))
                
        end_time = time.time()
        epoch_time = end_time - start_time
        print('epoch{} took {}s'.format(i, epoch_time))
        # TODO save the model


if __name__ == "__main__":
    main()