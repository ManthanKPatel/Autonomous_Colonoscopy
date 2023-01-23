import os

def formateImageNamesColon(pathLeft, pathRight):
	for filename in os.listdir(left):
		print("Renaming: ", filename)
		os.rename(os.path.join(pathLeft, filename), os.path.join(pathLeft, filename.replace("LMB_L", "")))	

	for filename in os.listdir(pathRight):
		print("Renaming: ", filename)
		os.rename(os.path.join(pathRight, filename), os.path.join(pathRight, filename.replace("LMB_R", "")))



left = os.path.join(os.getcwd(), "dataset/simulator/Case1/left")
right = os.path.join(os.getcwd(), "dataset/simulator/Case1/right")
formateImageNamesColon(left, right)



