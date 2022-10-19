from skimage import img_as_ubyte
from skimage.util import random_noise

def make_noise(type, img):
  '''
    type: gaussian | localvar | poisson | salt | pepper | s&p | speckle 
  '''
  if type == None:
    return img

  return img_as_ubyte(random_noise(img, mode=type))

