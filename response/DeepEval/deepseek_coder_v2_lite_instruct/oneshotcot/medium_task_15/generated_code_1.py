import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate, Multiply, AveragePooling2D, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Initial convolutional layer with batch normalization and ReLU activation
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Compress feature maps using global average pooling
    gap = GlobalAveragePooling2D()(x)
    
    # Two fully connected layers adjusting dimensions to match the channels of the initial features
    fc1 = Dense(units=64, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Reshape the output to match the size of the initial feature maps
    fc2_reshaped = keras.layers.Reshape((1, 1, 64))(fc2)
    
    # Multiply with the initial feature maps to generate weighted feature maps
    weighted_features = Multiply()([x, fc2_reshaped])
    
    # Concatenate the weighted feature maps with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Reduce dimensionality and downsample the feature using 1x1 convolution and average pooling
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and return the model
model = dl_model()