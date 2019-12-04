from mrcnn.rob_utils import *

config = ROBConfig()
config.display()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

dataset_train = ROBDataSet()
dataset_train.init_dataset()
dataset_train.prepare()

dataset_test = ROBDataSet()
dataset_test.init_dataset(ifTest=True, start_idx=5001, end_idx=6500)
dataset_test.prepare()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
	"mrcnn_bbox", "mrcnn_mask"])
model.train(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE,
	epochs=2, layers='heads')

