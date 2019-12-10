class ROBConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "rob"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    # IMAGES_PER_GPU = 8
    # Default is 2
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 4 class labels

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128
    # Default is 800 x 1024

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # DEFAULT: RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    # Default is 200

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100
    # Default is 1000
    STEPS_PER_EPOCH = int(5561/(GPU_COUNT*IMAGES_PER_GPU))

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 5

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.6
  
# Run these lines in the co-lab cell where this is imported:  
# config = ROBConfig()
# config.display()


class InferenceConfig(ROBConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1