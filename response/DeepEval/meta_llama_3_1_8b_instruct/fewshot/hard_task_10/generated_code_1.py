import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    conv2_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_path = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv2_path)
    conv4_path = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv3_path)
    
    adding_layer = Add()([conv1_path, conv4_path])
    
    # Apply 1x1 convolution to align output dimensions with input image's channel
    conv5_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(adding_layer)
    
    # Concatenate main path and branch path
    branch_path = input_layer
    merged_layer = Concatenate()([conv5_path, branch_path])
    
    # Apply two fully connected layers for classification
    flatten_layer = Flatten()(merged_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model