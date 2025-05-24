from skimage.feature import local_binary_pattern, hog
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import cv2

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_type='hog', pixel_per_cell=(2,2), block_per_cell=(2,2)):
        self.feature_type = feature_type
        self.pixel_per_cell = pixel_per_cell
        self.block_per_cell = block_per_cell

    def transform(self, X, y=None, visual=False):
        if self.feature_type == 'hog':
            if visual:
                featured, img_featured = hog(X[0], 
                                          orientations=8, 
                                          pixels_per_cell=self.pixel_per_cell,
                                          cells_per_block=self.block_per_cell, 
                                          visualize=visual)
                img_featured = np.expand_dims(img_featured, axis=-1)
                return img_featured, np.array(y)
            else:
                featured = hog(X[0], 
                             orientations=8, 
                             pixels_per_cell=self.pixel_per_cell,
                             cells_per_block=self.block_per_cell, 
                             visualize=visual)
                return np.array([featured]), np.array(y)
            
        elif self.feature_type == 'lbp':
            featured = local_binary_pattern(X[0], P=8, R=1, method='uniform').flatten()
            return np.array([featured]), np.array(y)
            
        elif self.feature_type == 'histogram':
            featured = np.histogram(X[0], bins=32, range=(0, 1))[0]
            return np.array([featured]), np.array(y)
        else:
            raise ValueError("Invalid feature type. Choose 'hog', 'lbp', or 'histogram'.")