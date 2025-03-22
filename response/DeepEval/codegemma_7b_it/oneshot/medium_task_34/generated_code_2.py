import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.initializers import glorot_uniform

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=1))(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=1))(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Part 2: Enhance Generalization
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=1))(max_pooling2)
    dropout = Dropout(rate=0.2)(conv3)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=1))(dropout)

    # Part 3: Upsampling and Restoration
    up_sampling1 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=1))(up_sampling1)
    skip_connection1 = Concatenate()([conv5, conv2])

    up_sampling2 = UpSampling2D(size=(2, 2))(skip_connection1)
    conv6 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=1))(up_sampling2)
    skip_connection2 = Concatenate()([conv6, conv1])

    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax', kernel_initializer=glorot_uniform(seed=1))(skip_connection2)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model