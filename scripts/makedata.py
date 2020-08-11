import os
import random

train_percent = 0.8
val_percent = 0.2
xmlfilepath = 'E:/workspace/CNN_PROJECT/darknet-master/data/car/VOCdevkit/VOC2012/Annotations'
txtsavepath = 'E:/workspace/CNN_PROJECT/darknet-master/data/car/VOCdevkit/VOC2012/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tr = int(num*train_percent)
tv = int(num*val_percent)
train = random.sample(list, tr)
val = random.sample(train, tr)

ftrain = open(txtsavepath + '/train.txt','w')
fval = open(txtsavepath + '/val.txt','w')

for i in list:
	name = total_xml[i][:-4] + '\n'
	if i in train:
		ftrain.write(name)
	else:
		fval.write(name)
	
ftrain.close()
fval.close()