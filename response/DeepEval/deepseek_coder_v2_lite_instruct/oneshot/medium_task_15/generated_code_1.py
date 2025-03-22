import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate, Multiply, AveragePooling2D, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(batch_norm1)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape the output to match the channels of the initial features
    reshaped_output = Dense(units=32, activation='relu')(dense2)
    
    # Multiply with the initial features to generate weighted feature maps
    weighted_feature_maps = Multiply()([batch_norm1, reshaped_output])
    
    # Concatenate with the input layer
    concatenated_output = Concatenate()([input_layer, weighted_feature_maps])
    
    # Reduce dimensionality and downsample the feature using 1x1 convolution and average pooling
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_output)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(conv2)
    
    # Flatten the result
    flattened_output = Flatten()(avg_pool)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model