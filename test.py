import argparse
import datetime
import json
import random
import time
from pathlib import Path

import datasets
import datasets.samplers as samplers
import matplotlib.pyplot as plt
import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# faster-rcnn
from configs.test_config import test_cfg
from utils_rcnn.draw_box_utils import draw_box, draw_box_rpn
from utils_rcnn.train_utils import create_model
from utils_rcnn.roi_header_util import RoIHeads


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    # parser.add_argument('--with_box_refine', default=True, action='store_true')
    # parser.add_argument('--two_stage', default=True, action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/root/dataset_clp/LPD_OPENDATASET_FINAL', type=str)
    # parser.add_argument('--coco_path', default='/root/dataset_clp/LPD_OPENDATASET_FINAL/test_parking', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/root/Deformable-DETR/result',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--faster_rcnn', default=True, action='store_true', help='use car proposal region')

    return parser

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, prob, boxes):
    plt.imshow(pil_img)
    print(pil_img.shape)
    colors = COLORS * 100
    # for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                          fill=False, color=c, linewidth=3))
        text = f'clp: {p:0.2f}'
        plt.gca().text(xmin, ymin, text, fontsize=6,
                       bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    
def infer(args):
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    faster_rcnn = create_model(num_classes=test_cfg.num_classes)
    faster_rcnn.to(args.device)
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cpu')
    faster_rcnn.load_state_dict(checkpoint['model'])

    # Build the model
    model, criterion, postprocessors = build_model(args)
    model.to(args.device)


    # Load the pretrained weights
    checkpoint = torch.load('/root/Deformable-DETR/exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage/checkpoint0044.pth', map_location='cuda')
    # checkpoint = torch.load('/root/Deformable-DETR/exps/r50_deformable_detr/checkpoint0099.pth', map_location='cuda')
    model.load_state_dict(checkpoint['model'])

    # Build the dataset
    dataset_test = build_dataset(image_set='test', args=args)

    # Create a dataloader
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                  drop_last=False, collate_fn=utils.collate_fn)

    # Ensure the model is in evaluation mode
    model.eval()
    faster_rcnn.eval()
    criterion.eval()
    
    # Loop over the test dataset
    for i, (images, targets) in enumerate(data_loader_test):
        # Move the images to the GPU
        # print(images.shape)
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
        # print(targets.shape)
        images = images.to(args.device)

        # faster_rcnn의 추론 진행
        predictions, bf_nms_boxes, pred_boxes = faster_rcnn(images)
        # Perform inference
        outputs, topk_coords_unact, outputs_coord = model(images, bf_nms_boxes)
        # loss_dict = criterion(outputs, targets)
        # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        boxes = [d.get('boxes', None) for d in results]
        # boxes = results['boxes']
        # Retrieve the scores

        probas = results[0]['scores']
        # Plot the results
        image, _ = dataset_test[i]  # 이미지와 레이블을 가져옵니다. 레이블은 필요하지 않으므로 무시합니다.
        pil_img = image.permute(1, 2, 0).numpy()  # PIL 이미지를 PyTorch 텐서로 변환한 다음, Numpy 배열로 변환합니다.
        
        plot_results(image.permute(1, 2, 0).numpy(), probas, boxes[0].cpu())
        # Save the plot
        plt.savefig("/root/Deformable-DETR/result/" + f"output_{i}_100.png")
        plt.close()

        keep = probas > 0.7

        bboxes_scaled = results[0]['boxes'][keep].cpu()
        # boxes = outputs['pred_boxes'][0, :]

        plot_results(pil_img, probas[keep], bboxes_scaled)

        # Save the plot
        plt.savefig("/root/Deformable-DETR/result/" + f"output_{i}.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR inference script', parents=[get_args_parser()])
    parser.add_argument('--model_path', type=str, help="Path to tAhe pretrained model")
    args = parser.parse_args()
    infer(args)
