# First we need to load all of the images specified in template.csv

import csv
import math
import os

image_filenames = []
csv_filename = "/content/drive/Shared drives/Self-Driving Cars Project/template.csv"
with open(csv_filename) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  line_count = 0
  for row in csv_reader:
      if line_count % 2 == 1:
        image_path = os.path.join('content/data-2019/test/', row[0][1:-2]+'_image.jpg')
        print(image_path)
        image_filenames.append(image_path)
      line_count += 1
  print(f'Processed {math.floor(line_count/2)} lines.')

# Second we need to run model.predict on all of these images.
# For each image, we need to calculate the centroid of the predicted mask to get the x and z values (y is always 0)
# Then we can compute r and theta for this image. Store these in a list.

# Finally we take our lists of r and theta and populate an output csv file to submit.