import os, csv, cv2
import numpy as np
import xml.etree.ElementTree as ET

def copyfile(source, des):
	img = cv2.imread(source)	
	print(source)
	print(des)
	cv2.imwrite(des, img)

def main():
	directoryXML = "raw/val2019/Annotation/"
	directoryIMG = "raw/val2019/Image/"
	path = os.path.join(os.getcwd(), directoryXML)

	count = 1
	data = []
	for folder in os.listdir(path):
		for file in os.listdir(os.path.join(path, folder)):
			img_list = os.listdir(os.path.join(os.getcwd(), directoryIMG + "/" + folder))
			xml_path = os.path.join(os.getcwd(), directoryXML + folder + "/" + file) 
			
			index = file.find(".")
			file = file[:index]

			if file + ".jpg" in img_list:
				xml_file = ET.parse(xml_path)
				xml = xml_file.getroot()

				object_xml = xml_file.find('object')

				if object_xml == None:

					file_name = str(count)
					label = []
					label.append(file_name)
					label.append('unspecified')
					data.append(label)
					source = os.path.join(os.getcwd(), directoryIMG + folder +"/"+ file + ".jpg")
					destination = os.path.join(os.getcwd(), "dataset/validation/unspecified/" + file_name + ".jpg")
					copyfile(source, destination)
					count += 1
					print('Image: ' + str(file_name) + ' - Polyp: Unspecified')
				else:
					file_name = str(count)
					name = object_xml.find('name').text
					difficult = object_xml.find('difficult').text
					truncated = object_xml.find('truncated').text
					bbox = object_xml.find('bndbox')
					xmin = bbox.find('xmin').text
					xmax = bbox.find('xmax').text
					ymin = bbox.find('ymin').text
					ymax = bbox.find('ymax').text
					label = []
					label.append(file_name)
					label.append(name)
					data.append(label)
					if name == 'adenomatous':
						source = os.path.join(os.getcwd(), directoryIMG + folder + "/" + file + ".jpg")
						destination = os.path.join(os.getcwd(), "dataset/validation/adenomatous/" + file_name + ".jpg")
						copyfile(source, destination)
					elif name == 'hyperplastic':
						source = os.path.join(os.getcwd(), directoryIMG + folder + "/" + file + ".jpg")
						destination = os.path.join(os.getcwd(), "dataset/validation/hyperplastic/" + file_name + ".jpg")
						copyfile(source, destination)
					else:
						print("Error")
					count += 1
					print('Image: ' + str(file_name) + ' - Polyp: ' + str(name))
			else:
				print("Image Not In DataBase")

	with open('validation.csv', 'w', encoding='UTF8') as file:
		writer = csv.writer(file)
		for label in data:
			writer.writerow(label)

if __name__ == '__main__':
	main()