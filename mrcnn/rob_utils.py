# utils fileimport sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

class ROBDataSet(utils.Dataset):
	def init_dataset(self, Root_Path='/Users/cadelau/Documents/Self-Driving/Project/rob535-fall-2019-task-1-image-classification/data-2019/', ifTest:bool=False, start_idx: int=0, end_idx: int=7573, ifInfere: bool=False):
		self.source = "ROB535"
		self.root_path = Root_Path
		self.train_img_list = glob(self.root_path+'trainval/*/*_image.jpg')

		if ifTest:
			self.test_img_list = self.train_img_list[start_idx:end_idx]
		if ifInfere:
			self.test_img_list = glob(self.root_path+'test/*/*_image.jpg')

		self.add_class(self.source, 1, 'label 0')
		self.add_class(self.source, 2, 'label 1')
		self.add_class(self.source, 3, 'label 2')

		if ifTest:
			for i, path in enumerate(self.test_img_list):
				self.add_image(self.source, i, path)
		else:
			for i, path in enumerate(self.train_img_list):
				self.add_image(self.source, i, path)

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info["path"]

	def load_mask(self, image_id):
		img_path = self.image_info[image_id]["path"]
		proj = np.fromfile(img_path.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
		proj.resize([3, 4])
		no_box = False

		try:
			bbox = np.fromfile(img_path.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
		except FileNotFoundError:
			no_box=True
			mask = np.empty([0, 0, 0])
			class_ids = np.empty([0], np.int32)
		if ~no_box:
			mask = np.zeros((1052, 1914, 1), dtype=np.uint8)
			bbox = bbox.reshape([-1, 11])

			label = np.array([class2label(bbox[0, -2])], np.int32)
			R = self.rot(bbox[0, 0:3])
			t = bbox[0, 3:6]
			sz = bbox[0, 6:9]
			vert_3D, _ = self.get_bbox(-sz / 2, sz / 2)
			vert_3D = R @ vert_3D + t[:, np.newaxis]
			vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
			vert_2D = vert_2D / vert_2D[2, :]
			min_x = min(vert_2D[0, :])
			max_x = max(vert_2D[0, :])
			min_y = min(vert_2D[0, :])
			max_y = max(vert_2D[0, :])
			points = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
			mask = cv2.fillConvexPoly(mask, np.int32(points), 1)
		return mask, label

	def rot(self, n):
		n = np.asarray(n).flatten()
		assert(n.size == 3)

		theta = np.linalg.norm(n)
		if theta:
			n /= theta
			K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

			return np.indentity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
		else:
			return np.indentity(3)

	def get_bbox(self, p0, p1):
		v = np.array([
			[p0[0], po[0], p0[0]. p0[0], p1[0], p1[0], p1[0], p1[0]],
			[p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
			[p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
		])
		e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],[7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)
		return v, e







