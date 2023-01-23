import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, auc, confusion_matrix

#threshold value
THRESH = 0.8

def accuracy(images, predictions):
	image_count = 0
	pred_count = 0
	for i in range(len(images)):
		label_index = None
		if (images[i][0] == 1):
			label_index = 0
		elif (images[i][1] == 1):
			label_index = 1
		elif(images[i][2] == 1):
			label_index = 2
		else:
			pass

		pred_index = None
		pred_index = np.argmax(predictions[i])

		image_count += 1
		if (label_index == pred_index):
			pred_count += 1

	print('Accuracy: ' + str(pred_count) + '/' + str(image_count))
	print('Accuracy Percentage: ' + str((pred_count / image_count) * 100.0))




def plot_roc_auc(images, predictions, classes):
	# initialize dictionaries and array
	fpr = dict()
	tpr = dict()
	roc_auc = np.zeros(3)

	plt.figure()
	colors = ['aqua', 'cornflowerblue']

	# for both classification tasks (categories 1 and 2) -> malignant and benign
	for i in range(3):
		# obtain ROC curve
		fpr[i], tpr[i], _ = roc_curve(images[:, i], predictions[:, i])
		# obtain ROC AUC curve
		roc_auc[i] = auc(fpr[i], tpr[i])
		# plot ROC curve
		plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of ' + classes[i])
	# get for score for category 3
	roc_auc[2] = np.average(roc_auc[:2])

	for i in range(3):
		print('Disease: ' + classes[i] + ' - ROC_AUC Score: ' + str(roc_auc[i]))

	# figure stuff
	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	plt.show()


def plot_confusion_matrix(images, predictions, classes):
	# obtain class probabilities
	predictions = (predictions>=THRESH)*1
	# obtain (unnormalized) confusion matrix
	cm = confusion_matrix(images.argmax(axis=1), predictions.argmax(axis=1))
	# normalize confusion matrix
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.0
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], '.2f'),
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()



def main():
	# import results from csv file
	result = pd.read_csv('results.csv')

	# import results into numpy arrays
	labels = result['label'].to_numpy()
	adenomatous = result['adenomatous'].to_numpy()
	hyperplastic = result['hyperplastic'].to_numpy()
	unspecified = result['unspecified'].to_numpy()

	# predictions matrix --> prediction[row][col]
	prediction = np.array([adenomatous, hyperplastic, unspecified]).transpose()
	
	x0 = []		# adenomatous
	x1 = []		# hyperplastic
	x2 = []		# unspecified
	for label in labels:
		if label == 0:
			x0.append(1)
			x1.append(0)
			x2.append(0)
		elif label == 1:
			x0.append(0)
			x1.append(1)
			x2.append(0)
		else:
			x0.append(0)
			x1.append(0)
			x2.append(1)

	# images labels matrix -->
	images = np.array([x0, x1, x2]).transpose()

	# skin lesions classes
	classes = ['adenomatous', 'hyperplastic', 'unspecified']

	# plot ROC curves and print scores
	plot_roc_auc(images, prediction, classes)

	# plot confusion matrix
	plot_confusion_matrix(images, prediction, classes)

	# output accracy of prediction
	accuracy(images, prediction)

if __name__ == '__main__':
	main()