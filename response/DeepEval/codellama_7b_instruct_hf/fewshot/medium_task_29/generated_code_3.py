import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16

# Load the pre-trained VGG-16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False


def dl_model():
    # Add three max pooling layers with varying window sizes
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(base_model.output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(base_model.output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(base_model.output)

    # Flatten the output of each pooling layer and concatenate them
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    merged = keras.layers.concatenate([flatten1, flatten2, flatten3])

    # Add two fully connected layers to produce the final classification results
    dense1 = Dense(64, activation='relu')(merged)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=base_model.input, outputs=dense2)

    return model