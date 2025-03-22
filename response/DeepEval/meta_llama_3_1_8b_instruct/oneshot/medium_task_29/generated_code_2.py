import keras
from keras.layers import Input, MaxPooling2D, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer with 32 filters, kernel size 3x3, and strides 1
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Max pooling layer with window size 1x1 and stride 1
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv)
    
    # Max pooling layer with window size 2x2 and stride 2
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(max_pooling_1)
    
    # Max pooling layer with window size 4x4 and stride 4
    max_pooling_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(max_pooling_2)
    
    # Concatenate the output of max pooling layers
    concat_output = Concatenate()([max_pooling_1, max_pooling_2, max_pooling_3])
    
    # Flatten the concatenated output into a one-dimensional vector
    flatten_layer = Flatten()(concat_output)
    
    # First fully connected layer with 128 units and ReLU activation
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer with 64 units and ReLU activation
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with 10 units (for 10 classes) and softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model