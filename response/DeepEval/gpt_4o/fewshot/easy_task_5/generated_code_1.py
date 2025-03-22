import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Reduce dimensionality using a 1x1 convolution
    reduce_dim = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Extract features using a 3x3 convolutional layer
    feature_extraction = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reduce_dim)
    
    # Restore dimensionality using another 1x1 convolution
    restore_dim = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(feature_extraction)
    
    # Flatten the output
    flatten_layer = Flatten()(restore_dim)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model