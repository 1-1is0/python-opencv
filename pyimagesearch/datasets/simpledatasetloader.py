import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
        
        # if the preprocessors are None, initializer them as an
        # empty list
        if self.preprocessors in None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose=-1):
        # initializer the list of features and labesl
        data = []
        labels = []
        
        # loop over the input image
        for (i, imagePath) in enumerate(imagePaths):
            # load the image nad extract the class label assuming
            # that our path has the following format
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            # check to see if our preprocessor are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updateing the data list followed by the labels
            data.append(image)
            labels.append(label)
            
            # show an update every verbose images     
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:

                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))
            

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
