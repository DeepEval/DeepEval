from PIL import Image, ImageEnhance

class ImageProcessor:
    def __init__(self):
        self.image = None

    def load_image(self, image_path):
        self.image = Image.open(image_path)

    def save_image(self, save_path):
        if self.image:
            self.image.save(save_path)

    def resize_image(self, width, height):
        if self.image:
            self.image = self.image.resize((width, height))

    def rotate_image(self, degrees):
        if self.image:
            self.image = self.image.rotate(degrees)

    def adjust_brightness(self, factor):
        if self.image:
            brightness = ImageEnhance.Brightness(self.image)
            self.image = brightness.enhance(factor)

if __name__ == "__main__":
    processor = ImageProcessor()

    # Test case 1
    processor.load_image("test.jpg")
    processor.resize_image(300, 300)
    processor.save_image("test2.jpg")
    print("Image resized and saved successfully!")

    # Test case 2
    processor.load_image("test.jpg")
    processor.rotate_image(90)
    processor.save_image("test3.jpg")
    print("Image rotated and saved successfully!")

    # Test case 3
    processor.load_image("test.jpg")
    processor.adjust_brightness(0.5)
    processor.save_image("test4.jpg")
    print("Image brightness adjusted and saved successfully!")