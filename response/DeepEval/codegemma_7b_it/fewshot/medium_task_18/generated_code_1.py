import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    # Block 2
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    conv2_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(max_pool1)

    # Feature concatenation
    concat = Concatenate()([conv1_1x1, conv1_3x3, conv1_5x5, conv2_1x1, conv2_3x3, conv2_5x5])

    # Block 3
    conv3_1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    conv3_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    conv3_5x5 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concat)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat)

    # Block 4
    conv4_1x1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    conv4_3x3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    conv4_5x5 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(max_pool3)

    # Feature concatenation
    concat = Concatenate()([conv3_1x1, conv3_3x3, conv3_5x5, conv4_1x1, conv4_3x3, conv4_5x5])

    # Global average pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(concat)

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(avg_pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model