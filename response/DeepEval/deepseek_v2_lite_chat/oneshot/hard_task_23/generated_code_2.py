import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, UpSampling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch: Focus on local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch1)
    
    # Second branch: Downsample, then upscale
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    zero_padding2 = ZeroPadding2D(padding=(1, 1))(branch2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(zero_padding2)
    up_conv2 = UpSampling2D(size=(2, 2))(conv2)
    
    # Third branch: Downsample, then upscale
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    zero_padding3 = ZeroPadding2D(padding=(1, 1))(branch3)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(zero_padding3)
    up_conv3 = UpSampling2D(size=(2, 2))(conv3)
    
    # Concatenate the outputs of the branches
    concatenated = Concatenate(axis=-1)([branch1, up_conv2, up_conv3, branch3])
    
    # Refine with another 1x1 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(conv4)
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and display the model
model = dl_model()
model.summary()