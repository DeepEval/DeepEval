import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)

    # Load pre-trained VGG16 model
    vgg16 = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')

    # Freeze pre-trained layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Add new layers for image classification
    x = Conv2D(64, (3, 3), activation='relu')(vgg16.output)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, vgg16.output])
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=vgg16.input, outputs=x)

    return model