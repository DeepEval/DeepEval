import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: First block: Three convolutional layers followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Step 3: Second block: Four convolutional layers followed by max pooling
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv7)

    # Step 4: Flattening the feature maps
    flatten_layer = Flatten()(max_pool2)
    
    # Step 5: Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Step 6: Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model