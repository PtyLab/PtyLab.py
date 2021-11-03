from unittest import TestCase
import imageio
from fracPy.utils.utils import shrink_image, pad_image

class Test(TestCase):
    def test_shrink_image(self):
        image = imageio.imread('imageio:camera.png')
        print(image.shape)
        for new_size in [12,25,26]:
            self.assertEqual(shrink_image(image, new_size).shape, (new_size, new_size))

        with self.assertRaises(ValueError):
            shrink_image(image, image.shape + 1)

    def test_pad_image(self):
        image = imageio.imread('imageio:camera.png')
        print(image.shape)
        for new_size in [12,25,26]:
            new_size = image.shape[-1] + new_size
            self.assertEqual(pad_image(image, new_size).shape, (new_size, new_size))

        with self.assertRaises(ValueError):
            pad_image(image, image.shape[-1]-1)
