import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten, Multiply, Concatenate, AveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    
    # Compression using Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Two fully connected layers
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    
    # Adjust dimensions to match the channels of initial features
    compressed_features = Dense(units=32, activation=None)(x)
    
    # Reshape compressed features to match the size of initial feature maps
    reshaped_features = keras.layers.Reshape((1, 1, 32))(compressed_features)
    
    # Multiply with initial features to generate weighted feature maps
    weighted_feature_maps = Multiply()([x, reshaped_features])
    
    # Concatenate weighted feature maps with the input layer
    concatenated = Concatenate()([input_layer, weighted_feature_maps])
    
    # Reduce dimensionality and downsample the feature using 1x1 convolution and average pooling
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    x = AveragePooling2D(pool_size=(8, 8))(x)
    
    # Flatten the result
    x = Flatten()(x)
    
    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model