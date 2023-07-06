import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from configs.test_config import test_cfg
from dataloader.coco_dataset import coco
from utils_rcnn.draw_box_utils import draw_box, draw_box_rpn
from utils_rcnn.train_utils import create_model
from utils_rcnn.roi_header_util import RoIHeads


def test():
    print(torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=test_cfg.num_classes)
    model.to(device)
    # model.cuda()
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # read class_indict
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_data_set = coco(test_cfg.data_root_dir, 'test', '', data_transform)
    category_index = test_data_set.class_to_coco_cat_id

    index_category = dict(zip(category_index.values(), category_index.keys()))

    original_img = Image.open(test_cfg.image_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # predictions, bf_nms_boxes = model(img.cuda())
        predictions, bf_nms_boxes, pred_boxes = model(img.cuda())
        predictions = predictions[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("No target detected!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 index_category,
                 thresh=0.3,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.savefig('/root/pytorch-faster-rcnn/output/output.jpg')
        plt.close()
        print(bf_nms_boxes.shape)
        # plt.show()
        bf_nms_boxes = bf_nms_boxes.to("cpu").numpy()
        draw_box_rpn(original_img,
                 bf_nms_boxes,
                 line_thickness=3)
        
        plt.imshow(original_img)
        plt.savefig('/root/pytorch-faster-rcnn/output/output_rpn.jpg')
        plt.close()
        
        draw_box_rpn(original_img,
                 pred_boxes[0],
                 line_thickness=3)
        
        plt.imshow(original_img)
        plt.savefig('/root/pytorch-faster-rcnn/output/output_pred.jpg')
        plt.close()


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    test()
