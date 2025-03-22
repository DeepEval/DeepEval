import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Add

def dl_model():
    """
    Deep learning model for image classification using the MNIST dataset.
    
    The model consists of a main path and a branch path. In the main path, the input first passes 
    through two <convolution,dropout> blocks to gradually increase the feature width. This is 
    followed by another convolutional layer to restore the number of channels. The branch path 
    connects directly to the input. The outputs from both paths are combined through an addition 
    operation. Finally,a flattening layer and a fully connected layer produce the final 
    classification probabilities.
    """
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    
    # Branch path
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main and branch paths
    add = Add()([conv3, branch])
    
    # Add max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(add)
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(max_pooling)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model