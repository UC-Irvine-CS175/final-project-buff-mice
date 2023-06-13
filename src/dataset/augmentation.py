'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class 
that takes a list of transformations and applies them in order, which 
can be chained together simply by defining a __call__ method for each class. 
'''
import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        print("JERERERERERE")
        print(img_array)
        mag = np.linalg.norm(img_array)
        if mag == 0:
            return img_array
        return img_array / mag

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self.height = resize_height
        self.width = resize_width
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        temp = np.resize(img, (self.width, self.height))
        # return torch.Tensor(temp.astype(np.float32))
        return temp

class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(image, (s1[1],s1[0]), interpolation = cv2.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img[y1:y2, x1:x2] = img_resize
            return img
        else:
            return img_resize

class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        return np.flipud(image)


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        return np.fliplr(image)


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        self.rotation = rotate

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270 degrees of rotation.
        Returns:
            np.ndarray
        '''
        
        # it is 90 and stuff or smth
        num_rots = self.rotation / 90
        temp = np.rot90(image, num_rots)
        return temp


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_height: int, output_width: int):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image):
        assert image.shape[0] >= self.output_width and image.shape[1] >- self.output_height, "too small to crop"
        
        # get the maximum allowed starting coordinate for the crop
        max_start_x = image.shape[0] - self.output_width
        max_start_y = image.shape[1] - self.output_height

        # start loc
        start_x = np.random.randint(max_start_x)
        start_y = np.random.randint(max_start_y)

        image = image[start_x :(start_x + self.output_width), start_y: (start_y + self.output_height)]
        return image
        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) < 3:
            new_image = image[:, :, np.newaxis] 
        else:
            new_image = image
        assert len(new_image.shape) == 3, "wrong dimensions buddy"
        cpy = new_image.copy()
        return torch.from_numpy(cpy).permute(2, 0, 1)

def main():
    """Driver function for testing the augmentations. Make sure the file paths work for you."""
    # load image using cv2
    img_key = 'P280_73668439105-F5_015_023_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    type(test_resize)

if __name__ == "__main__":
    main()