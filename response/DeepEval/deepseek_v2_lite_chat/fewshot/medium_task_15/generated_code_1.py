import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    # Batch normalization and ReLU activation
    batchnorm1 = BatchNormalization()(conv)
    act1 = Activation('relu')(batchnorm1)
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(act1)
    
    # Two fully connected layers for dimension adjustment
    dense1 = Dense(units=512, activation='relu')(avg_pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Reshape dense layers output for multiplication with initial features
    reshape = Reshape((10, 10, 1))(dense2)
    # Weighted feature maps
    weighted_feature_maps = Concatenate()([reshape, act1])
    
    # Reduce dimensionality and downsample
    conv_downsample = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(weighted_feature_maps)
    avg_pool_downsample = GlobalAveragePooling2D()(conv_downsample)
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(avg_pool_downsample)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()