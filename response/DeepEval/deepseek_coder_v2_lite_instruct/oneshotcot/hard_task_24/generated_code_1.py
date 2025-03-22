import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Branch 2: Sequential downsampling and upsampling
    branch2_down = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_down)
    branch2_up = UpSampling2D(size=(2, 2))(branch2)
    
    # Branch 3: Sequential downsampling and upsampling
    branch3_down = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_down)
    branch3_up = UpSampling2D(size=(2, 2))(branch3)
    
    # Concatenate the outputs of all branches
    concatenated = Concatenate()([branch1, branch2_up, branch3_up])
    
    # 1x1 convolutional layer to fuse the features
    fused = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Batch normalization
    batch_norm = BatchNormalization()(fused)
    
    # Flatten the output
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model