import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D
from keras.models import Model

def dl_model():
    
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu_activation = ReLU()(batch_norm)
    
    # Global average pooling to compress feature maps
    global_avg_pool = GlobalAveragePooling2D()(relu_activation)
    
    # Fully connected layers
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Reshape the output of the fully connected layers to match initial features
    reshaped_dense = Reshape((1, 1, 32))(dense2)
    
    # Multiply with the initial feature maps to create weighted feature maps
    weighted_features = Multiply()([relu_activation, reshaped_dense])
    
    # Concatenate the weighted features with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Reduce dimensionality and downsample using 1x1 convolution and average pooling
    conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)
    downsampled = AveragePooling2D(pool_size=(2, 2))(conv1x1)
    
    # Final fully connected layer for classification
    final_flatten = GlobalAveragePooling2D()(downsampled)
    output_layer = Dense(units=10, activation='softmax')(final_flatten)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example of creating the model
model = dl_model()
model.summary()