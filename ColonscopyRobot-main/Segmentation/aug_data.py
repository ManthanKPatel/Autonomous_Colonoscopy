import cv2
import sys, os
import numpy as np
import argparse, json
from json import JSONEncoder


class classification:
	def __init__(self, idx, image_id, category_id, name, bbox, seg):
		self.id = idx
		self.image_id = image_id
		self.category_id = category_id
		self.name = name
		self.bbox = bbox
		self.segmentation = seg


class image:
	def __init__(self, img, data):
		self.width = img['width']
		self.height = img['height']
		self.id = img['id']
		self.filename = img['file_name']
		self.classification = []
		for anno in data['annotations']:
			if anno['image_id'] == self.id:
				idx = anno['id']
				image_id = anno['image_id']
				category_id = anno['category_id']
				name = data['categories'][int(anno['category_id'])]['name']
				bbox = anno['bbox']
				seg = anno['segmentation']
				# debug formate
				#print('id: ', idx)
				#print('image_id: ', image_id)
				#print('category_id: ', category_id)
				#print('name: ', name)
				#print('bbox: ', bbox)
				#print('seg: ', seg)
				#print()
				self.classification.append(classification(idx, image_id, category_id, name, bbox, seg))

		
class JsonObject:
	def __init__(self, file):
		json_dic = json.load(file)
		self.data = []
		for img in json_dic['images']:
			self.data.append(image(img, json_dic))
		self.classes = []
		for disease in json_dic['categories']:
			self.classes.append(disease['name'])


def formateFilename(name):
	# remove directory '/'
	while name.find("/") != -1:
		pos = name.find("/") 
		name = name.replace(name[0:pos+1], "")
	# remove extension
	pos = name.find(".")
	return name[:pos]


def formateFilenameWithExtension(name):
	# remove directory '/'
	while name.find("/") != -1:
		pos = name.find("/") 
		name = name.replace(name[0:pos+1], "")
	return name


def writeMask(args, json_data):
	root = os.path.join(os.getcwd(), args.source)
	for img in json_data:
		for anno in img.classification:
			path = os.path.join(root, anno.name + '/mask')
			mask = np.zeros((img.height, img.width), dtype=np.int32)
			points = np.array(anno.segmentation, dtype=np.int32).reshape((-1,2))
			points = points.reshape((-1, 1, 2))
			cv2.fillPoly(mask, [points], 255)
			filename = formateFilename(img.filename)
			cv2.imwrite(os.path.join(path, filename + ".png") , mask)

def copyImages(args, json_data):
	root = os.path.join(os.getcwd(), args.source)
	for img in json_data:
		for anno in img.classification:
			filename = formateFilename(img.filename)
			filename_extension = formateFilenameWithExtension(img.filename)
			image_path = os.path.join(os.getcwd(), args.results + "/images/" + filename_extension)
			print(image_path)
			image_copy = cv2.imread(image_path)
			path = os.path.join(root, anno.name + "/image")
			cv2.imwrite(os.path.join(path, filename + ".jpg"), image_copy)



def generateDirectories(args, classes):
	root = os.path.join(os.getcwd(), args.source)
	if os.path.exists(root):
		print("File folder /", args.source, " already exists! Exiting...")
		print("Try Rename or Deleting folder")
		sys.exit()
	os.mkdir(root)
	for disease in classes:
		disease_folder = os.path.join(root, disease)
		disease_img_folder = os.path.join(disease_folder, "image")
		disease_mask_folder = os.path.join(disease_folder, "mask")
		os.mkdir(disease_folder)
		os.mkdir(disease_img_folder)
		os.mkdir(disease_mask_folder)


def main(args):
	json_file = os.path.join(args.results, 'result.json')
	# encode json into object
	with open(json_file) as file:
		json_object = JsonObject(file)
		json_data = json_object.data
		json_class = json_object.classes
	generateDirectories(args, json_class)
	writeMask(args, json_data)
	copyImages(args, json_data)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Provide Argument to build training data')
	parser.add_argument('--results', type=str, help='.../path/results')
	parser.add_argument('--source', type=str, help='.../path/soure/training...')
	args = parser.parse_args()
	# check if there is 5 arguments
	if len(sys.argv) < 5:
		parser.print_help()
		sys.exit()
	# check if directory exists
	if os.path.exists(os.path.join(os.getcwd(), args.results)) is False:
		print("Check Formated Results Path ...")
		sys.exit()
	main(args)
