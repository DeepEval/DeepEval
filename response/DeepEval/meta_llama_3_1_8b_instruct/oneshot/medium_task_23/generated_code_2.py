import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    path2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path2_conv1)
    path2_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path2_conv2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    path3_conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path3_conv1)
    path3_conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path3_conv2)
    path3_conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path3_conv3)
    path3_conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path3_conv4)

    # Path 4: Average pooling followed by 1x1 convolution
    path4_avgpool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(path4_avgpool)

    # Concatenate the outputs of these paths
    output_tensor = Concatenate()([path1_conv, path2_conv3, path3_conv5, path4_conv])

    # Apply batch normalization
    batch_norm = BatchNormalization()(output_tensor)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model