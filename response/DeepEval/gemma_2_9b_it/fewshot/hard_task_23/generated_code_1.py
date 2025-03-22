import keras
from keras.layers import Input, Conv2D, AveragePooling2D, TransposedConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch 1: Local feature extraction
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2_1)

    # Branch 2: Downsampling and upsampling with average pooling
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv3_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    upsample3_1 = TransposedConv2D(filters=32, kernel_size=(2, 2), strides=(2, 2))(conv3_1) 

    # Branch 3: Downsampling and upsampling with average pooling
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv4_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool2)
    upsample4_1 = TransposedConv2D(filters=32, kernel_size=(2, 2), strides=(2, 2))(conv4_1) 

    # Concatenate branch outputs
    combined = Concatenate()([conv2_2, upsample3_1, upsample4_1])

    # Refine with 1x1 convolution
    conv_final = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(combined)

    # Flatten and output layer
    flatten = Flatten()(conv_final)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model