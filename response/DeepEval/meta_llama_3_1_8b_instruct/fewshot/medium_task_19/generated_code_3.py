import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch: dimensionality reduction using 1x1 convolution
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second branch: extracting features using 1x1 and 3x3 convolution
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Third branch: capturing larger spatial information using 1x1 and 5x5 convolution
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    # Fourth branch: downsampling using 3x3 max pooling and further processing using 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    branch4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    
    # Concatenating the outputs of all branches
    adding_layer = Add()([branch1, branch2, branch3, branch4])
    
    # Flattening the features
    flatten_layer = Flatten()(adding_layer)
    
    # Using two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model