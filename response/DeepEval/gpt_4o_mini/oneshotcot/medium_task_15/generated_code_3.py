import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from keras.layers import Dense, Reshape, Multiply, Concatenate, AveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Step 1: Convolutional layer with Batch Normalization and ReLU
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Step 2: Global Average Pooling
    gap = GlobalAveragePooling2D()(relu)
    
    # Step 3: Two fully connected layers
    dense1 = Dense(units=32, activation='relu')(gap)  # Adjusted to be the same as channels of initial features
    dense2 = Dense(units=32, activation='relu')(dense1)  # Same as above
    
    # Step 4: Reshape to match the size of the initial features
    reshaped_output = Reshape((1, 1, 32))(dense2)  # Reshape to (1, 1, 32) to match feature maps
    
    # Step 5: Multiply with the initial features to generate weighted feature maps
    weighted_features = Multiply()([relu, reshaped_output])
    
    # Step 6: Concatenate the weighted features with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Step 7: Dimensionality reduction and downsampling using 1x1 convolution
    conv_reduced = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Step 8: Average Pooling
    pooled_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_reduced)
    
    # Step 9: Fully connected layer for output
    flatten_layer = GlobalAveragePooling2D()(pooled_output)  # Flatten before the final Dense layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model