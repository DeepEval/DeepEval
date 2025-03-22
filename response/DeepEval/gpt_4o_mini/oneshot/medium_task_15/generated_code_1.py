import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, AveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)
    
    # Initial convolution layer followed by Batch Normalization and ReLU activation
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    # Global Average Pooling to compress feature maps
    gap = GlobalAveragePooling2D()(relu)
    
    # Two fully connected layers that match the dimensions of the initial feature maps
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape to match the size of the initial features
    reshaped = Reshape((1, 1, 32))(dense2)
    
    # Multiply reshaped output with the initial feature maps
    weighted_features = Multiply()([relu, reshaped])
    
    # Concatenate the weighted features with the input layer
    concatenated = keras.layers.Concatenate()([input_layer, weighted_features])
    
    # Dimensionality reduction and downsampling using 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(concatenated)
    pooled = AveragePooling2D(pool_size=(2, 2))(conv1x1)
    
    # Final fully connected layer for output
    flatten_layer = keras.layers.Flatten()(pooled)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model