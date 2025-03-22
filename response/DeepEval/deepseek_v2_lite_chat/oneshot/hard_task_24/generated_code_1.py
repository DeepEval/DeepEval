import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Branch 1: 3x3 convolutional layer
    conv1_branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    conv1_branch1 = BatchNormalization()(conv1_branch1)
    
    # Branch 2: Max Pooling -> 3x3 Convolution -> UpSampling -> Restore size
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(input_layer)
    conv2_branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pooling1)
    upscale2 = UpSampling2D(size=(2, 2))(conv2_branch2)
    conv2_branch2 = ZeroPadding2D(padding=(1, 1))(upscale2)
    
    # Branch 3: Max Pooling -> 3x3 Convolution -> UpSampling -> Restore size
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(input_layer)
    conv3_branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pooling2)
    upsample3 = UpSampling2D(size=(2, 2))(conv3_branch3)
    conv3_branch3 = ZeroPadding2D(padding=(1, 1))(upsample3)
    
    # Concatenate all branches
    concat = Concatenate(axis=-1)([conv1_branch1, conv2_branch2, conv3_branch3])
    
    # Final 1x1 convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    conv4 = BatchNormalization()(conv4)
    
    # Flatten and feed into fully connected layers
    flatten = Flatten()(conv4)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])