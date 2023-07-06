import math
import os
from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import argparse
import test
import torchvision
from pathlib import Path
from models.deformable_detr import build
from models.deformable_detr import DeformableDETR
from util.misc import NestedTensor
torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'N/A', 'licenses'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    # T.Resize((1008,800)),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return b

# def plot_results(pil_img, prob, boxes, output_file):
#     plt.figure(figsize=(16,10))
#     plt.imshow(pil_img)
#     ax = plt.gca()
#     colors = COLORS * 100
#     for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color=c, linewidth=3))
#         cl = p.argmax()
#         text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
#         ax.text(xmin, ymin, text, fontsize=15,
#                 bbox=dict(facecolor='yellow', alpha=0.5))
#     plt.axis('off')
#     plt.savefig(output_file)

def plot_results(pil_img, prob, boxes, output_file):
    plt.imshow(pil_img)
    # print(pil_img.shape)
    colors = COLORS * 100
    # for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                          fill=False, color=c, linewidth=3))
        text = f'clp: {p:0.2f}'
        plt.gca().text(xmin, ymin, text, fontsize=6,
                       bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if output_file.endsWith('.jpg') or output_file.endsWith('.png') : plt.savefig(output_file)
    plt.close()

# from models.backbone import build_backbone
# command ="--dataset_file coco --with_box_refine --two_stage --faster_rcnn --batch_size 1 --eval --no_aux_loss --coco_path /root/dataset_clp/ --output_dir result --resume /root/Deformable-DETR/exps/r50_deformable_detr/checkpoint0099.pth --num_workers=8"
command ="--dataset_file coco --batch_size 1 --eval --no_aux_loss --coco_path /root/dataset_clp/ --output_dir result --resume /root/Deformable-DETR/exps/r50_deformable_detr/checkpoint0099.pth --num_workers=8"

parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[test.get_args_parser()])
args = parser.parse_args(command.split())
print(args)
if args.two_stage == False : 
    two_stage = False
else : two_stage = True


dataset_path = '/root/dataset_clp/LPD_OPENDATASET_FINAL/test_parking/test_none'
test_dataset = os.listdir(dataset_path)

# 모델 생성
device = torch.device('cuda')

# FasterRCNN
# from utils_rcnn.train_utils import create_model
# from configs.test_config import test_cfg
# faster_rcnn = create_model(num_classes=test_cfg.num_classes)
# faster_rcnn.to(args.device)
# weights = test_cfg.model_weights

# checkpoint = torch.load(weights, map_location='cpu')
# faster_rcnn.load_state_dict(checkpoint['model'])
# faster_rcnn.eval()
# predictions, bf_nms_boxes, pred_boxes = faster_rcnn(img)

# Deformable DETR
model, criterion, postprocessors = build(args)
if two_stage :
    state_dict = torch.load('/root/Deformable-DETR/exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/checkpoint0044.pth')
else : 
    state_dict = torch.load('/root/Deformable-DETR/exps/r50_deformable_detr/checkpoint0099.pth')
    
model.load_state_dict(state_dict['model'])
model.to(device)

error_file = []
for idx, input_file in enumerate(test_dataset) :
    file = os.path.join(dataset_path, input_file)
    print(file)
    try :
        im = Image.open(file)
    except :
        error_file.append(file)
    
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img = img.to(device)  # 입력 데이터를 CUDA 디바이스로 이동

    # propagate through the model
    outputs, tp, infer_coord = model(img, None)
    
    b, c, w, h = img.shape
    orig_traget_size = torch.tensor([[w, h]]).to(device)
    results = postprocessors['bbox'](outputs, orig_traget_size)
    boxes = [d.get('boxes', None) for d in results]
    probas = results[0]['scores']
    keep = probas > 0.3
    bboxes_scaled = results[0]['boxes'][keep].cpu()
    
    # 여러개가 나오는 방법이였음..
    # keep only predictions with 0.7+ confidence
    # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values > 0.9
    # best = keep.max()
    # # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
    
    if two_stage :
        output_file = os.path.join('/root/Deformable-DETR/result_parkinglot', input_file)
    else : 
        output_file = os.path.join('/root/Deformable-DETR/result_parkinglot_normal', input_file)
    
    print(bboxes_scaled)
    img = img.squeeze().permute(1,2,0).detach().cpu()
    plot_results(img, probas[keep], bboxes_scaled, output_file)

print(error_file)