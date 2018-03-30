import os
import json
from PIL import Image
import random
import shutil

#Looks at whatever is inside this json_list
json_list = ['parr_marks_annotations'] #list of all the annotations we currently have
val_ratio = 0.25 # 25% of these images will go validation instead
training_count = 0
val_count = 0

for json_name in json_list:
	print('now looking at '+json_name)
	with open(json_name + '.json') as json_data:
		d = json.load(json_data)
	for i in range(len(d)):
		if len(d[i]['annotations']) == 0:
			print 'no annotations'
		else:
			output_dir = os.getcwd() + '/parr_dataset/train/'
			training_count += 1
			if random.random() < val_ratio:
				output_dir = os.getcwd() + '/parr_dataset/val/'
				training_count -= 1
				val_count+= 1
			fn = str(d[i]['filename'])
			shutil.copy(os.getcwd() + '/' + fn,output_dir+ 'images/') #Copy from source to dst
			fnbase, ext = os.path.splitext(fn)
			with open(output_dir+ 'labels/' + fnbase + '.txt', 'w') as fp:
				for j in range(len(d[i]['annotations'])):
					l = d[i]['annotations'][j]['x'] 
					t = d[i]['annotations'][j]['y']
					r = l + d[i]['annotations'][j]['width']
					b = t + d[i]['annotations'][j]['height']
					type = 'Car'
					truncated = 0
					occluded  = 3
					alpha  = 0
					tail = '0 0 0 0 0 0 0 0'
					label = type + ' ' +   \
					str(truncated) + ' ' +  \
					str(occluded)  + ' ' +  \
					str(alpha)     + ' ' +  \
					str(l) + ' ' + str(t) + ' ' + str(r) + ' ' + str(b) + ' ' + tail
					fp.write(label + '\n')

print(training_count)
print(val_count)

