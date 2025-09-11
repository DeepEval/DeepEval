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
        Use Image util in PIL to open a image
        :param image_path: str, path of image that is to be loaded
        """
        try:
            self.image = Image.open(image_path)
            return self.image
        except FileNotFoundError:
            return "Error: File not found."
        except Exception as e:
            return str(e)

    def save_image(self):
        """
        Save image to a path if image has opened
        """
        if self.image:
            self.image.save(f'test2.jpg')
            return "Image saved successfully."
        else:
            return "No image opened."

    def resize_image(self, width, height):
        """
        Resize the image if image has opened.
        :param width: int, the target width of image
        :param height: int, the target height of image
        """
        if self.image:
            resized_image = self.image.resize((width, height))
            return resized_image
        else:
            return "No image opened."

    def rotate_image(self, degrees):
        """
        Rotate image if image has opened.
        :param degrees: float, the degrees that the image will be rotated
        """
        if self.image:
            rotated_image = self.image.rotate(degrees)
            return rotated_image
        else:
            return "No image opened."

    def adjust_brightness(self, factor):
        """
        Adjust the brightness of image if image has opened.
        :param factor: float, brightness of an image. A factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
        """
        if self.image:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(factor)
            return self.image
        else:
            return "No image opened."

# Test cases
if __name__ == "__main__":
    processor = ImageProcessor()

    # Test load_image
    print(processor.load_image('test.jpg'))

    # Test save_image
    processor.save_image()

    # Test resize_image
    print(processor.resize_image(300, 300))

    # Test rotate_image
    print(processor.rotate_image(90))

    # Test adjust_brightness
    processor.adjust_brightness(0.5)