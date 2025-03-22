import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Add, Activation, BatchNormalization
from keras.models import Model
from keras.applications import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Channel Attention
    # Path 1: Global Average Pooling
    avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(avg_pooling)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Path 2: Global Max Pooling
    max_pooling = GlobalMaxPooling2D()(input_layer)
    dense3 = Dense(units=128, activation='relu')(max_pooling)
    dense4 = Dense(units=10, activation='softmax')(dense3)

    # Concatenate and add the outputs
    block1_output = Add()([dense1, dense2, dense3, dense4])

    # Block 2: Spatial Attention
    # Average Pooling
    avg_pooling_block2 = GlobalAveragePooling2D()(block1_output)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(avg_pooling_block2)
    relu1 = Activation('relu')(conv1)

    # Max Pooling
    max_pooling_block2 = GlobalMaxPooling2D()(block1_output)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(max_pooling_block2)
    relu2 = Activation('relu')(conv2)

    # Concatenate and add the outputs
    block2_output = Add()([relu1, relu2])

    # Flatten and add the outputs
    flatten_output = Flatten()(block2_output)
    dense5 = Dense(units=128, activation='relu')(flatten_output)
    dense6 = Dense(units=10, activation='softmax')(dense5)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense6)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model