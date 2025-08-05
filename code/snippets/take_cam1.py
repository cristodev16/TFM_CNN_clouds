import pandas as pd 
import pickle

# Load images (real path might be different)
images = pd.read_pickle("/home/csanchezmoreno/tfm/data/imageset.pickle")

# Select camera 1
images_cam1 = images[:,:,:,0,:]

# Store reduced set of images (real path might be different)
with open("/home/csanchezmoreno/tfm/data/imageset_cam1.pickle", "wb") as f:
    pickle.dump(images_cam1, f)