import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():

    # Load the VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the VGG16 model layers
    for layer in vgg.layers:
        layer.trainable = False

    # Add a new input layer
    input_layer = Input(shape=(32, 32, 3))

    # Add a convolutional layer followed by a max-pooling layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Add a concatenate layer to merge the output features from VGG16 and the new input layer
    concatenate_layer = Concatenate()([max_pooling, vgg.output])

    # Add a batch normalization layer
    batch_norm = BatchNormalization()(concatenate_layer)

    # Flatten the output of the batch normalization layer
    flatten_layer = Flatten()(batch_norm)

    # Add two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model