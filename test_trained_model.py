#trained model testing on training images, update "VARIABLES TO BE CONFIGURED" section with your own paths

#import the usual suspects
import numpy as np
import cv2
import os
import glob

#import TF and object detection packages
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


#### VARIABLES TO BE CONFIGURED ####
IMAGES_PATH = glob.glob('H:/object_detection-runescape_rocks/images/*.jpg')
PATH_TO_SAVED_MODEL = 'H:/object_detection-runescape_rocks/exported-models/faster_rcnn_resnet50_v1/saved_model/'
PATH_TO_LABELS =      'H:/object_detection-runescape_rocks/annotations/label_map.pbtxt'
N_rand = 5 #number of images to test/show
####             ####           ####



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
#configure GPU(s) if any
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('Loading model...')

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

#pick N random images to show
IMAGES_PATH = np.random.choice(IMAGES_PATH,N_rand)
print(IMAGES_PATH)
#loop over random images, test out the model
for image_path in IMAGES_PATH:
    print('Running inference for {}... '.format(image_path), end='')
    print('\nPress \'q\' to move to next image')

    #load image, force "IMREAD_COLOR", loads RGB image, convert to BGR for model eval 
    image_np = cv2.imread(image_path,1)[...,::-1]

    #make copy of loaded image for plotting of boxes, convert BGR -> RGB for display
    image_np_with_detections = image_np.copy()[...,::-1]
    
    # Convert image to tensor using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    #add "batch" dim.
    input_tensor = input_tensor[tf.newaxis, ...]
    #call prediction model
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)    

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
           min_score_thresh=.40,
          agnostic_mode=False)

    cv2.imshow('window',image_np_with_detections)
    #wait until user presses 'q', move on to next image
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()