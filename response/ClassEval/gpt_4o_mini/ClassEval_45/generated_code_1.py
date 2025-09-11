from PIL import Image, ImageEnhance

class ImageProcessor:
    """
    This is a class to process images, including loading, saving, resizing, rotating, and adjusting the brightness of images.
    """

    def __init__(self):
        """
        Initialize self.image
        """
        self.image = None

    def load_image(self, image_path):
        """
        Use Image util in PIL to open an image.
        :param image_path: str, path of image that is to be opened.
        """
        self.image = Image.open(image_path)
        return self.image

    def save_image(self, save_path):
        """
        Save image to a path if image has been opened.
        :param save_path: str, the path that the image will be saved.
        """
        if self.image:
            self.image.save(save_path)
            return True
        return False

    def resize_image(self, width, height):
        """
        Resize the image if image has been opened.
        :param width: int, the target width of image.
        :param height: int, the target height of image.
        """
        if self.image:
            self.image = self.image.resize((width, height))
            return self.image.size
        return None

    def rotate_image(self, degrees):
        """
        Rotate image if image has been opened.
        :param degrees: float, the degrees that the image will be rotated.
        """
        if self.image:
            self.image = self.image.rotate(degrees)
            return self.image
        return None

    def adjust_brightness(self, factor):
        """
        Adjust the brightness of image if image has been opened.
        :param factor: float, brightness of an image. A factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
        """
        if self.image:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(factor)
            return self.image
        return None

# Test cases to validate the functionality
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # Test load_image
    loaded_image = processor.load_image('test.jpg')  # Replace with a valid path
    print(f'Loaded image: {loaded_image}')

    # Test save_image
    save_success = processor.save_image('test2.jpg')  # Replace with a valid path
    print(f'Image saved: {save_success}')

    # Test resize_image
    new_size = processor.resize_image(300, 300)
    print(f'Resized image to: {new_size}')

    # Test rotate_image
    rotated_image = processor.rotate_image(90)
    print(f'Rotated image: {rotated_image}')

    # Test adjust_brightness
    brightened_image = processor.adjust_brightness(1.5)
    print(f'Adjusted brightness: {brightened_image}')