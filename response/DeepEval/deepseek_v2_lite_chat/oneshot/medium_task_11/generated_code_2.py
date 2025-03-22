import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Parallel feature extraction paths
    def average_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        return dense2
    
    def max_path(input_tensor):
        max_pool = GlobalMaxPooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(max_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        return dense2
    
    # Outputs from parallel paths
    avg_output = average_path(conv1)
    max_output = max_path(conv1)
    
    # Add and activation
    add_layer = Concatenate()([avg_output, max_output])
    activation = Dense(units=2)(add_layer)
    activation = Activation('softmax')(activation)  # Generate channel attention weights
    
    # Processed features
    channel_pool1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='same')(conv1)
    spatial_pool1 = GlobalAveragePooling2D()(conv1)
    
    # Concatenate channel and spatial features
    concat_layer = Concatenate()([channel_pool1, spatial_pool1])
    
    # Fully connected layers
    dense3 = Dense(units=128, activation='relu')(concat_layer)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()