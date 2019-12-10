from mrcnn.rob_utils import *
from mrcnn.rob_config import *
import scipy
import skimage.color
import skimage.io
import skimage.transform


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = "/logs/rob201912**T****/mask_rcnn_rob_00##.h5"
model_path = "/content/drive/Shared drives/Self-Driving Cars Project/h5_files/Trial_2/mask_rcnn_rob_0001.h5"
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


labels = []
print(len(dataset_test.test_img_list))
for image_path in dataset_test.test_img_list:
  # image = dataset_test.load_image(image_id)
  image = skimage.io.imread(image_path)
  image,_,_,_,_ = utils.resize_image(
      image,
      min_dim=config.IMAGE_MIN_DIM,
      min_scale=config.IMAGE_MIN_SCALE,
      max_dim=config.IMAGE_MAX_DIM,
      mode=config.IMAGE_RESIZE_MODE)
  results = model.detect([image], verbose=1)
  r = results[0]
  rois = r['rois']
  largest_index = -1
  largest_area = -1
  index = 0
  for roi in rois:
    area = (roi[2]-roi[0]) * (roi[3] - roi[1])
    if area > largest_area:
      largest_area = area
      largest_index = index
    index = index + 1
  if largest_index == -1:
    largest_class_id = 0
  else:
    largest_class_id = r['class_ids'][largest_index] - 1 # Didn't forget!
  labels.append(largest_class_id)

generate_output_csv(labels)
