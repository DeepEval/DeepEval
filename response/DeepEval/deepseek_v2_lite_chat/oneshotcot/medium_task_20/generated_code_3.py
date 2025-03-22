import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution, 3x3 convolution, and max pooling
    def first_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        return max_pool
    
    path1 = first_path(input_tensor=input_layer)
    
    # Second path: two stacked 3x3 convolutions following a 1x1 convolution
    def second_path(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        return conv4
    
    path2 = second_path(path1)
    
    # Third path: single 3x3 convolution following a 1x1 convolution
    def third_path(input_tensor):
        conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv5
    
    path3 = third_path(path2)
    
    # Fourth path: max pooling, 1x1 convolution, and 3x3 convolution
    def fourth_path(input_tensor):
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pool)
        return conv6
    
    path4 = fourth_path(path3)
    
    # Concatenate the outputs of the four paths
    concat_layer = Concatenate()([path1, path2, path3, path4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten = Flatten()(batch_norm)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()