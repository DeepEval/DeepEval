import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the VGG16 model as a pre-trained feature extractor
    vgg16 = VGG16(include_top=False, input_shape=(32, 32, 3))

    # Freeze the layers of the pre-trained model
    for layer in vgg16.layers:
        layer.trainable = False

    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Apply 1x1 max pooling with a stride of 1
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)

    # Apply 2x2 max pooling with a stride of 2
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Apply 4x4 max pooling with a stride of 4
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten the output of the last pooling layer
    x = Flatten()(x)

    # Concatenate the outputs of the three pooling layers
    x = keras.layers.concatenate([x, x, x])

    # Add two fully connected layers to produce the output
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=x)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model