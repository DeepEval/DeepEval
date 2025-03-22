import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    # Block 1
    gap = GlobalAveragePooling2D()(input_layer)
    dense_layer1 = Dense(units=input_layer.shape[3], activation='relu')(gap)
    dense_layer2 = Dense(units=input_layer.shape[3], activation='sigmoid')(dense_layer1)
    reshape_layer = Reshape((input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]))(dense_layer2)
    weighted_feature_output = multiply([input_layer, reshape_layer])

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_feature_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Branch from Block 1
    branch_output = max_pooling

    # Fusion
    combined_output = add([branch_output, max_pooling])

    # Classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model