import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, UpSampling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch: local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Second branch: downsampling and upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    
    # Third branch: downsampling and upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    
    # Concatenate outputs
    concat_layer = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer for refining the output
    refined_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    
    # Flatten the output and apply a fully connected layer for classification
    flatten_layer = Flatten()(refined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model