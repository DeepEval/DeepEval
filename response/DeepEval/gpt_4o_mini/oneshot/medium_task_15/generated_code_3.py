import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, AveragePooling2D
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Batch normalization and ReLU activation
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)
    
    # Fully connected layers
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape to match the number of channels in the original feature map
    reshaped = Reshape((1, 1, 32))(dense2)
    
    # Multiply with the initial feature maps
    weighted_features = Multiply()([relu, reshaped])
    
    # Concatenate the weighted features with the input layer
    concatenated = keras.layers.Concatenate()([input_layer, weighted_features])
    
    # Dimensionality reduction with 1x1 convolution
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated)
    
    # Average pooling to downsample
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv1x1)
    
    # Final fully connected layer for classification
    flatten_layer = GlobalAveragePooling2D()(avg_pool)  # Flatten before the dense layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model