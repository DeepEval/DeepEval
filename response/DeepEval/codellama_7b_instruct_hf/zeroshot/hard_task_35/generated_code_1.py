import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, Concatenate
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the VGG16 model
    for layer in vgg_model.layers:
        layer.trainable = False

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the block
    def block(x):
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        # Fully connected layers
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        # Reshape the weights
        x = Flatten()(x)
        return x

    # Define the model
    inputs = Input(shape=input_shape)
    branch1 = block(vgg_model(inputs))
    branch2 = block(vgg_model(inputs))
    outputs = Concatenate()([branch1, branch2])
    outputs = Flatten()(outputs)
    outputs = Dense(10, activation='softmax')(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model