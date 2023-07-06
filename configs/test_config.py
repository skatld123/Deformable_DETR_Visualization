class Config:
    model_weights = "/root/pytorch-faster-rcnn/result/resnet50_fpn-model-84-mAp-0.6008944783196632.pth"
    image_path = "/root/dataset_clp/dataset_car/test/leaf_3_jpg.rf.21fc128d135ef99f915796f70a379297.jpg"
    gpu_id = '0, 1'
    num_classes = 3 + 1
    data_root_dir = "/root/dataset_clp/dataset_car/"


test_cfg = Config()
