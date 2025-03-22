import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolution layer
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    bn_1 = BatchNormalization()(conv_1)
    act_1 = Activation('relu')(bn_1)

    # Global average pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2))(act_1)

    # Fully connected layer 1
    fc_1 = Dense(units=128, activation='relu')(avg_pool)

    # Fully connected layer 2
    fc_2 = Dense(units=64, activation='relu')(fc_1)

    # Reshape for feature map multiplication
    fc_reshape = Flatten()(fc_2)

    # Feature map multiplication
    mul_feature = Dense(units=32 * 32 * 3)(fc_reshape) # multiply by the number of elements in the feature maps
    mul_feature = Reshape([32, 32, 3])(mul_feature)

    # Initial feature maps
    initial_feature = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the weighted feature maps with the initial feature maps
    concat_layer = Concatenate()([mul_feature, initial_feature])

    # Second convolutional layer with average pooling
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)

    # Flatten and fully connected layers
    flatten = Flatten()(avg_pool_2)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)

    return model