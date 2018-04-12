import time
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2


def identifyImage(model_file, pretrained_model, image, steelhead_boolean):
   caffe.set_mode_gpu()

   net = caffe.Net(model_file,pretrained_model, caffe.TEST )


   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   transformer.set_transpose('data', (2,0,1))
   transformer.set_raw_scale('data', 255)
   transformer.set_channel_swap('data', (2,1,0))

   BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = net.blobs['data'].data[...].shape

   print ('The input size for the network is: (' + \
   str(BATCH_SIZE), str(CHANNELS), str(HEIGHT), str(WIDTH) + \
   ') (batch size, channels, height, width)')

   img = cv2.imread(image)

   img = cv2.resize(img, (1280,720), 0, 0)

   data = transformer.preprocess('data', img.astype('float16')/255)

   net.blobs['data'].data[...] = data
   start = time.time()
   print("Now with " + str(model_file))
   if(steelhead_boolean):
      bounding_boxes = net.forward()['bbox-list-class0'][0]
   else:
      bounding_boxes = net.forward()['bbox-list'][0]
   end = (time.time() - start)*1000

   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   overlay = img.copy()

   for bbox in bounding_boxes:
       if  bbox.sum() > 0:
           cv2.rectangle(overlay, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), -1)

   img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

   cv2.putText(img, "Inference time: %dms per frame" % end, (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
   cv2.imshow('Project Pisces',img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def identifyVideo(model_file, pretrained_model, image, steelhead_boolean):
   caffe.set_mode_gpu()

   net = caffe.Net(model_file,pretrained_model, caffe.TEST )


   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   transformer.set_transpose('data', (2,0,1))
   transformer.set_raw_scale('data', 255)
   transformer.set_channel_swap('data', (2,1,0))

   BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = net.blobs['data'].data[...].shape

   print ('The input size for the network is: (' + \
   str(BATCH_SIZE), str(CHANNELS), str(HEIGHT), str(WIDTH) + \
   ') (batch size, channels, height, width)')


   img = cv2.resize(image, (1280,720), 0, 0)

   data = transformer.preprocess('data', img.astype('float16')/255)

   net.blobs['data'].data[...] = data
   start = time.time()
   if(steelhead_boolean):
      bounding_boxes = net.forward()['bbox-list-class0'][0]
   else:
      bounding_boxes = net.forward()['bbox-list'][0]
   end = (time.time() - start)*1000

   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   overlay = img.copy()

   for bbox in bounding_boxes:
       if  bbox.sum() > 0:
           cv2.rectangle(overlay, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), -1)

   img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

   cv2.putText(img, "Inference time: %dms per frame" % end, (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def detectImage(model_file, pretrained_model, image):
   caffe.set_mode_gpu()

   net = caffe.Net(model_file,pretrained_model, caffe.TEST )

   transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
   transformer.set_transpose('data', (2,0,1))
   transformer.set_raw_scale('data', 255)
   transformer.set_channel_swap('data', (2,1,0))

   BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = net.blobs['data'].data[...].shape

   print ('The input size for the network is: (' + \
   str(BATCH_SIZE), str(CHANNELS), str(HEIGHT), str(WIDTH) + \
   ') (batch size, channels, height, width)')

   img = cv2.imread(image)

   img = cv2.resize(img, (1280,720), 0, 0)

   data = transformer.preprocess('data', img.astype('float16')/255)

   net.blobs['data'].data[...] = data
   bounding_boxes = net.forward()['bbox-list'][0]
   return bounding_boxes

