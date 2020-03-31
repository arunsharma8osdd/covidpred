from __future__ import division

import tensorflow as tf
import numpy as np
import os
import cv2
import time
import sys

## All varieties directories names

all_varities = ['Normal','covid_19','non_covid_19','Pneumonia','TB']

for vrty in all_varities:

    # Training images path (will be used for class index reading)
    train_p = './train_augmented_rotated_140'	

    # Validation set images directory path    
    test_p = './test_augmented_rotated_140/'+vrty	


    # Saved model directory path
    model_p = 'Model_rotated_140_degree_24_epochs'    

    model_p2 = model_p+'/trained_model.meta'

    pred_class_arr = []

    start = time.time()

    try:
        import os

        train_path = train_p

        if not os.path.exists(train_path):
            print("No such directory")
            raise Exception

         # Path of testing images

        dir_path = test_p

        if not os.path.exists(dir_path):
            print("No such directory")
            raise Exception
	img_count=1;

        # Walk though all testing images one by one
        for root, dirs, files in os.walk(dir_path):
            for name in files:

                image_path = name
                filename = dir_path +'/' +image_path

		# Image size and number of channels (3 for RGB images)
                image_size=256
                num_channels=3
                images = []

                if os.path.exists(filename):

                    # Reading the image using OpenCV
                    image = cv2.imread(filename)

                    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                    images.append(image)
                    images = np.array(images, dtype=np.uint8)
                    images = images.astype('float32')
                    images = np.multiply(images, 1.0/255.0) 

                    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                    x_batch = images.reshape(1, image_size,image_size,num_channels)

                    # Let us restore the saved model 
                    sess = tf.Session()

                    # Step-1: Recreate the network graph. At this step only graph is created.
                    #saver = tf.train.import_meta_graph('Model_augmented_CLoDSA_gamma_250_epochs_BS_256/trained_model.meta')
                    saver = tf.train.import_meta_graph(model_p2)

                    # Step-2: Now let's load the weights saved using the restore method.
                    #saver.restore(sess, tf.train.latest_checkpoint('./Model_augmented_CLoDSA_gamma_250_epochs_BS_256/'))
                    saver.restore(sess, tf.train.latest_checkpoint(model_p))

                    # Accessing the default graph which we have restored
                    graph = tf.get_default_graph()

                    # Now, let's get hold of the op that we can be processed to get the output.
                    # In the original network y_pred is the tensor that is the prediction of the network
                    y_pred = graph.get_tensor_by_name("y_pred:0")

                    ## Let's feed the images to the input placeholders
                    x= graph.get_tensor_by_name("x:0") 
                    y_true = graph.get_tensor_by_name("y_true:0") 
                    y_test_images = np.zeros((1, len(os.listdir(train_path)))) 


                    # Creating the feed_dict that is required to be fed to calculate y_pred 
                    feed_dict_testing = {x: x_batch, y_true: y_test_images}
                    result=sess.run(y_pred, feed_dict=feed_dict_testing)

                    # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                    #print "ID:"+image_path, "Prob:", result

                    # Convert np.array to list
                    a = result[0].tolist()
                    r=0

                    # Finding the maximum of all outputs
                    max1 = max(a)
                    index1 = a.index(max1)
                    predicted_class = None

                    # Walk through directory to find the label of the predicted output
                    count = 0
                    for root, dirs, files in os.walk(train_path):
                        for name in dirs:
                            if count==index1:
                                predicted_class = name
                            count+=1

                    # If the maximum confidence output is largest of all by a big margin then
                    # print the class 
                    for i in a:
                        if i!=max1:
                            if max1-i<i:
                                r=1                           
                    if r == 0:
                        pred_class_arr.append(predicted_class)
			img_count=img_count+1
                    else:
                        pred_class_arr.append(predicted_class)
			img_count=img_count+1
                # If file does not exist
                else:
                    print("File does not exist")
    except Exception as e:
        print("Exception:",e)

    all_varities = ['Normal','covid_19','non_covid_19','Pneumonia','TB']

    d = {}
    for item in pred_class_arr:
        if item in d:
            d[item] = d.get(item)+1
        else:
            d[item] = 1

    total_test_images = len(pred_class_arr)

    print("\n\nTotal test images \(" + vrty + "\): ", total_test_images)

    print("Prediction\tProportion (%)\n")

    for vrrt in all_varities:
        if vrrt in d:
            vrr = d[vrrt]
            #print vrr
            acc = round((vrr*100/total_test_images),2)
            print vrrt+": {}%".format(acc)
        else:
            print(vrrt+": 0%")

# Calculate execution time
end = time.time()
dur = end-start
print("")
if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
