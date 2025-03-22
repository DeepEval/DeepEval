import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Branch 1: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(conv1)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
    
    # Concatenate the outputs from the three branches
    concat_layer = Concatenate()([conv3, conv6])
    # 1x1 convolution to adjust the output dimensions to match the input image's channel size
    conv7 = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(concat_layer)
    
    # Fusion of the main path and the branch
    fused_output = keras.layers.Add()([conv3, conv7])
    # Batch normalization and flattening
    bath_norm = BatchNormalization()(fused_output)
    flatten_layer = Flatten()(bath_norm)
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model