import keras
from keras.layers import Input, Lambda, Flatten, Concatenate, Permute, Reshape, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=1))(input_layer)
    x = Concatenate()(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Flatten()(x)

    # Second block
    x = Reshape((32, 32, 3))(x)
    x = Permute((2, 1, 3))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    # Third block
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Flatten()(x)

    # Fourth block
    x = Add()([x, input_layer])
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    return model