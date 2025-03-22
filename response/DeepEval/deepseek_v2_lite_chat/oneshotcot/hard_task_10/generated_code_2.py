import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction path 1: 1x1 convolution
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Feature extraction path 2
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(7, 1), padding='valid', activation='relu')(input_layer)
    
    # Concatenate the outputs of the two paths
    concat_layer = Concatenate()([conv1_1, conv1_2, conv1_3])
    
    # 1x1 convolution to match the output dimension with the input image's channel
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    
    # Branch directly connected to the input
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of the main path and the branch
    add_layer = Add()([conv2_1, branch])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(add_layer)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model