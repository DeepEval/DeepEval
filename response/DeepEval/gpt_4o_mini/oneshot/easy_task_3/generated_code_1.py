import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First feature extraction block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    # Second feature extraction block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(max_pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    # Third feature extraction block
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(max_pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3_2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_3)

    # Fourth feature extraction block
    conv4_1 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(max_pool3)
    conv4_2 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4_2)
    max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_3)

    # Flatten the feature maps
    flatten_layer = Flatten()(max_pool4)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model