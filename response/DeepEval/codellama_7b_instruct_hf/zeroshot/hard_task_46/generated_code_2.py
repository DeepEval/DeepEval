import keras
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate, Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    x = Input(shape=input_shape)
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x1 = SeparableConv2D(1, (1, 1), activation='relu')(x1)
    x2 = SeparableConv2D(3, (3, 3), activation='relu')(x2)
    x3 = SeparableConv2D(5, (5, 5), activation='relu')(x3)
    x = Concatenate()([x1, x2, x3])

    # Define the second block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x1 = Conv2D(3, (3, 3), activation='relu')(x1)
    x2 = Conv2D(3, (3, 3), activation='relu')(x2)
    x3 = MaxPooling2D((3, 3))(x3)
    x = Concatenate()([x1, x2, x3])

    # Define the output layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=x, outputs=x)

    return model