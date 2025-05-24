from skimage.feature import local_binary_pattern,hog
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type='hog',pixel_per_cell=(2,2),block_per_cell=(1,1)):
        self.feature_type = feature_type
        self.pixel_per_cell = pixel_per_cell
        self.block_per_cell = block_per_cell

    def transform(self, X, y=None,visual=False):
        if self.feature_type == 'hog':
            print(f"hog shape: {X.shape} batching")
           
            if visual:
               pack_featured = np.array([hog(img, orientations=8, pixels_per_cell=self.pixel_per_cell,
                                  cells_per_block=self.block_per_cell, visualize=visual) for img in X])
               featured,img_featured = zip(*pack_featured)
               print(f"featured shape: {featured.shape}")
               img_featured = np.expand_dims(img_featured, axis=-1)
               return img_featured,np.array(y)
            else:    
                featured = np.array([hog(img, orientations=8, pixels_per_cell=self.pixel_per_cell,
                                    cells_per_block=self.block_per_cell, visualize=visual) for img in X])
                print(f"featured shape: {featured.shape}")
                return featured, np.array(y)
            
        elif self.feature_type == 'lbp':
            return np.array([local_binary_pattern(img, P=8, R=1, method='uniform').flatten()
                             for img in X]), np.array(y)
        elif self.feature_type == 'histogram':
            return np.array([np.histogram(img, bins=32, range=(0, 1))[0] for img in X]), np.array(y)
        else:
            raise ValueError("Invalid feature type. Choose 'hog', 'lbp', or 'histogram'.")