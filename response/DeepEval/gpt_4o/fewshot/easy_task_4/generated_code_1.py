import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: three convolutional layers followed by max pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_3)

    # Second block: four convolutional layers followed by max pooling
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1)
    conv2_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_2)
    conv2_4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_3)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_4)

    # Flatten the feature maps
    flatten_layer = Flatten()(pool2)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model