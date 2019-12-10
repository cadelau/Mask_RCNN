#### PART 1
# First we need to load all of the images specified in template.csv
# %%capture
import csv
import math
import os

image_pathnames = []
csv_filename = "/content/drive/Shared drives/Self-Driving Cars Project/template.csv"
with open(csv_filename) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  line_count = 0
  for row in csv_reader:
      if line_count % 2 == 1:
        image_path = os.path.join('/content/data-2019/test/', row[0][0:-2]+'_image.jpg')
        print(image_path)
        image_pathnames.append(image_path)
      line_count += 1
  print(f'Processed {math.floor(line_count/2)} lines.')

#### PART 2: STILL NEED TO DETERMINE HOW TO CALC X,Y, and Z!!!
# Second we need to run model.predict on all of these images.
# For each image, we need to calculate the centroid of the predicted mask to get the x and z values (y is always 0)
# Then we can compute r and theta for this image. Store these in a list.
# Second we need to run model.predict on all of these images.
# For each image, we need to calculate the centroid of the predicted mask to get the x and z values (y is always 0)
# Then we can compute r and theta for this image. Store these in a list.
# import rob_test.py
import scipy
import skimage.color
import skimage.io
import skimage.transform
class InferenceConfig(ROBConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = "/logs/rob201912**T****/mask_rcnn_rob_00##.h5"
model_path = "/content/drive/Shared drives/Self-Driving Cars Project/h5_files/Trial_1/mask_rcnn_rob_0001.h5"
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

r_vals = []
theta_vals = []
for image_path in image_pathnames:
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
  largest_roi = roi[0]
  index = 0
  # selecting largest block assuming it is the closest.
  for roi in rois:
    area = (roi[2]-roi[0]) * (roi[3] - roi[1])
    if area > largest_area:
      largest_area = area
      largest_index = index
      largest_roi = roi
    index = index + 1
  
  # if for some reason our algorithm doesn't detect anything, pick default x,y,z values
  if largest_index == -1:
    x = 0
    y = 0
    z = 25
  else:
    x = (largest_roi[2] + largest_roi[0])/2 + 512 # shift (0,0) from top left to center
    y = (largest_roi[3] + largest_roi[1])/2 + 512 # shift (0,0) from top left to center
    z = 25
  r_val = math.sqrt(x*x + y*y + z*z)
  theta_val = math.degrees(math.atan(x/z))
  r_vals.append(r_val)
  theta_vals.append(theta_val)
  print(r_val)


#### PART 3
# Finally we take our lists of r and theta and populate an output csv file to submit.

# Set up csv rows list and specify filename
csv_rows = [['guid/image/axis', 'value']]
csv_filename = 'task2_results.csv'

for i in range(len(r_vals)):
  r_val = r_vals[i]
  theta_val = theta_vals[i]
  split_pathname = image_pathnames[i].split('/')
  guid = split_pathname[4] + '/' + split_pathname[5][0:4]
  guid_r = guid + '/r'
  guid_theta = guid + '/theta'
  csv_rows.append([guid_r, r_val])
  csv_rows.append([guid_theta, theta_val])

# Write rows list to csv file
with open(csv_filename, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_rows)

print(len(csv_rows), ' rows written to the ', csv_filename, ' csv file.')