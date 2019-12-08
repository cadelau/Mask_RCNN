from mrcnn.rob_utils import *
from mrcnn.rob_config import *


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Test on a random image
image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# results = model.detect([original_image], verbose=1)

# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                             dataset_test.class_names, r['scores'], ax=get_ax())
results = model.detect([original_image], verbose=1)

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
# print(largest_index)
largest_roi = np.reshape(r['rois'][largest_index,:], (1,4))
largest_mask = np.reshape(r['masks'][:,:,largest_index], (128, 128, 1))
largest_class_id = np.array([r['class_ids'][largest_index]])
largest_score = np.array([r['scores'][largest_index]])
visualize.display_instances(original_image, largest_roi, largest_mask, largest_class_id, 
                            dataset_test.class_names, largest_score, ax=get_ax())


#### SUBTRACT 1 FROM LABEL for CSV file!!!!