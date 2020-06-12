############## cleaning given xml file###########
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
 

class KangarooDataset(Dataset):
############## find bounding box and extract it ##############
	def extract_boxes(self,filename):
		tree = ElementTree.parse(filename)
		root = tree.getroot()
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
############## preparing Dataset#################
	def load_dataset(self,dataset_dir,is_train = True):
		self.add_class("dataset", 1, "kangaroo")
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		for filename in listdir(images_dir):
			image_id = filename[:-4]
			if image_id in ['00090']:
				continue
			if is_train and int(image_id) >= 150:
				continue
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
	def load_mask(self,image_id):
		info = self.image_info[image_id]
		path = info['annotation']
		boxes,width,height = self.extract_boxes(path)
		mask = zeros([height,width,len(boxes)], dtype='uint8')
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			mask[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('kangaroo'))
		return mask, asarray(class_ids, dtype='int32')
 
	def image_refrence(self,image_id):
		info = self.image_info[image_id]
		return info['path']


###########################################
####TRAINING THE MODEL ON WEIGHTS##########
###########################################

class KangarooConfig(Config):
	NAME = "Kangaroo_cfg"
	NUM_CLASSES = 1+1
	STEPS_PER_EPOCH = 131



train_set = KangarooDataset()
train_set.load_dataset('kangaroo-master', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
 
# test/val set
test_set = KangarooDataset()
test_set.load_dataset('kangaroo-master', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# # load an image
# image_id = 1
# image = train_set.load_image(image_id)
# print(image.shape)

# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)

###########################################################
########### to show images with bounding box ##############
###########################################################

# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	image = train_set.load_image(i)
# 	pyplot.imshow(image)
# 	# plot all masks
# 	mask, _ = train_set.load_mask(i)
# 	for j in range(mask.shape[2]):
# 		pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# # show the figure
# pyplot.show()

###################################################################################
############# debugging to show all th dictionary with respecteed paths############
###################################################################################

# for image_id in train_set.image_ids:
# 	# load image info
# 	info = train_set.image_info[image_id]
# 	# display on the console
# 	print(info)
#####################################################################
########## to display your image with display instane################
#####################################################################

# image_id = 0
# # load the image
# image = train_set.load_image(image_id)
# # load the masks and the class ids
# mask, class_ids = train_set.load_mask(image_id)
# # extract bounding boxes from the masks
# bbox = extract_bboxes(mask)
# # display image with masks and bounding boxes
# display_instances(image, bbox, mask, class_ids, train_set.class_names)


config = KangarooConfig()
config.display()

## using model
model = MaskRCNN(mode = 'training', model_dir = './',config = config)
#load weightss
model.load_weights('mask_rcnn_coco.h5',by_name = True,exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
## training starts
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')