import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Second block
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Align the spatial dimensions to match the input before addition
    # Upsample the feature map back to original dimensions before addition
    upsampled_pool2 = keras.layers.UpSampling2D(size=(4, 4))(max_pooling2)

    # Add the input layer and upsampled output of the second block
    add = Add()([input_layer, upsampled_pool2])

    # Flatten the resulting feature map and add a fully connected layer for classification
    flatten_layer = Flatten()(add)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model