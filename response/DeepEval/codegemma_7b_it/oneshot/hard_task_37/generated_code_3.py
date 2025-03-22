import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    conv1_branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1_branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_branch1)
    conv2_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1_branch1)
    max_pooling2_branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_branch1)
    conv3_branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2_branch1)

    # Branch 2
    conv1_branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1_branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_branch2)
    conv2_branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(max_pooling1_branch2)
    max_pooling2_branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_branch2)
    conv3_branch2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(max_pooling2_branch2)

    # Parallel branch
    conv_parallel = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Addition operation
    addition_output = Add()([conv3_branch1, conv3_branch2, conv_parallel])

    # Concatenation operation
    concat_output = Concatenate()([addition_output, max_pooling2_branch1])

    # Fully connected layers
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model