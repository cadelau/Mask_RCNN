
from mrcnn.rob_utils import *

config = ROBConfig()
config.display()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Train Dataset
dataset_train = ROBDataSet()
dataset_train.init_dataset()
dataset_train.prepare()

# Validation Dataset
dataset_val = ROBDataSet()
dataset_val.init_dataset(ifTest=True, start_idx=5001, end_idx=6500)
dataset_val.prepare()

# Test Dataset
dataset_test = ROBDataSet()
dataset_test.init_dataset(ifInfere=True)
dataset_test.prepare()

# model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
# 	"mrcnn_bbox", "mrcnn_mask"])
# model.train(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE,
# 	epochs=2, layers='heads')
BEST_WEIGHTS = "/content/drive/Shared drives/Self-Driving Cars Project/h5_files/Trial_2/mask_rcnn_rob_0002.h5"
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(BEST_WEIGHTS, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=2, layers='heads')
