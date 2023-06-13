'''
Image Augmentations for use with the BPSMouseDataset class and PyTorch Transforms
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
        # Normalizing uint numpy arrays representing images to a floating point
        # range between 0 and 1 brings the pixel values of the image to a common
        # scale that is compatible with most deep learning models.
        # Additionally, normalizing the pixel values can help to reduce the effects
        # of differences in illumination and contrast across different images, which
        # can be beneficial for model training. To normalize, we divide each pixel
        # value by the maximum value of the uint16 data type.

        # Normalize array values between 0 - 1
        img_array = img_array / np.iinfo(np.uint16).max

        # Conversion of uint16 -> float32
        img_normalized = img_array.astype(np.float32)

        # img_normalized = img_float / np.max(img_float)  # 65535.0

        return img_normalized

class RescaleBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Rescale the array values between -1 and 1
        """
        img_array = img_array / np.iinfo(np.uint16).max
        img_float = img_array.astype(np.float32)
        img_rescaled = img_float * 2 - 1
        return img_rescaled

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self.resize_width = resize_width
        self.resize_height = resize_height
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        img_resized = cv2.resize(img, (self.resize_width, self.resize_height))
        return img_resized

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        img = image.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        #img_tensor = torch.from_numpy(image).unsqueeze(0)
        # image = image.transpose((2, 0, 1))
        return img_tensor

def main():
    """"""
    # load image using cv2
    img_key = 'P280_73668439105-F5_015_023_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    type(test_resize)

    # test_augmentations_img = BPSAugmentations(img_array)

    # test_output_normalize= test_augmentations_img.normalize_bps()
    # test_output_resize = test_augmentations_img.resize_bps(500, 500)
    # test_output_vflip = test_augmentations_img.v_flip()
    # test_output_hflip = test_augmentations_img.h_flip()

    # #FIXME - Fix the rotate function to include superimage after
    # #test_output_rotate = test_augmentations_img.rotate(45)

    # # Output zoom increases or decreases the overall size
    # # of the image--must be followed up with resize in order to make
    # # sure all images are the same for PyTorch DataLoader
    # test_output_zoom = test_augmentations_img.zoom(3)
    # test_zoom_resize = BPSAugmentations(test_output_zoom).resize_bps(100,100)
    # print(test_output_zoom.shape)
    # print(test_zoom_resize.shape)





    # # Attempt to save

    # # Show as tensor
    # # plt.imshow(img_array[0][:][:])

    # # Show as np.array
    # plt.imshow(img_array)
    # plt.savefig('augmentations_b4_test.png')
    # # Show as np.array
    # plt.imshow(test_output_hflip)
    # plt.savefig('augmentations_hflip.png')
    # plt.imshow(test_output_vflip)
    # plt.savefig('augmentations_vflip.png')
    # plt.imshow(test_output_resize)
    # plt.savefig('augmentations_resize.png')
    
    # plt.imshow(test_zoom_resize)
    # plt.savefig('augmentations_after_zoom_resize.png')

if __name__ == "__main__":
    main()



### json file with a list of aug: object like structure -> image, : List of aougment preformed, and their description
### 



# Remove the black void:
# 1. Find dimensions to zoom in to
# 2. Call Zoom in function
# 3. Scale back to 64x64