from PIL import Image, ImageEnhance

class ImageProcessor:
    """
    This is a class to process image, including loading, saving, resizing, rotating, and adjusting the brightness of images.
    """

    def __init__(self):
        """
        Initialize self.image
        """
        self.image = None

    def load_image(self, image_path):
        """
        Use Image util in PIL to open an image
        :param image_path: str, path of image that is to be
        """
        self.image = Image.open(image_path)
        return self.image

    def save_image(self, save_path):
        """
        Save image to a path if image has opened
        :param save_path: str, the path that the image will be saved
        """
        if self.image is not None:
            self.image.save(save_path)
            return f"Image saved to {save_path}"
        else:
            return "No image to save."

    def resize_image(self, width, height):
        """
        Resize the image if image has opened.
        :param width: int, the target width of image
        :param height: int, the target height of image
        """
        if self.image is not None:
            self.image = self.image.resize((width, height))
            return self.image
        else:
            return "No image to resize."

    def rotate_image(self, degrees):
        """
        Rotate image if image has opened
        :param degrees: float, the degrees that the image will be rotated
        """
        if self.image is not None:
            self.image = self.image.rotate(degrees)
            return self.image
        else:
            return "No image to rotate."

    def adjust_brightness(self, factor):
        """
        Adjust the brightness of image if image has opened.
        :param factor: float, brightness of an image. A factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
        """
        if self.image is not None:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(factor)
            return self.image
        else:
            return "No image to adjust brightness."

# Test cases for each method
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # Test load_image
    image = processor.load_image('test.jpg')
    print(image)

    # Test save_image
    result = processor.save_image('test2.jpg')
    print(result)

    # Test resize_image
    resized_image = processor.resize_image(300, 300)
    print(resized_image.size if resized_image else "Resize failed.")

    # Test rotate_image
    rotated_image = processor.rotate_image(90)
    print(rotated_image.size if rotated_image else "Rotate failed.")

    # Test adjust_brightness
    brightened_image = processor.adjust_brightness(0.5)
    print(brightened_image if brightened_image else "Brightness adjustment failed.")