from datasets import ListDataset
from utils import *
import pdb

def evaluate(model, dataloader, iou_thres, classes, device):
    # pdb.set_trace()
    model.eval()
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for i, (images, targets) in enumerate(
            tqdm.tqdm(dataloader, desc="evaluating;", total=len(dataloader))):
        images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        # Extract labels
        for target in targets:
            labels += target['labels'].tolist()
        # imgs = [Variable(img.type(Tensor), requires_grad=False) for img in imgs]
        # print(labels)
        with torch.no_grad():
            outputs = model(images)
            # outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    # pdb.set_trace()
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    # print('true_positive length', len(true_positives))
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({classes[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    return precision, recall, AP, f1, ap_class
