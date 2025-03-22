import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape, Flatten
from keras.models import Model

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Three 3x3 convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    
    # Max pooling layer
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Branch path
    # Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers to generate channel weights
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)  # Sigmoid to output weights
    
    # Reshape weights to match the input shape for multiplication
    reshape_weights = Reshape((1, 1, 32))(dense2)
    
    # Multiply weights with the input (this assumes the input is downsampled to 32 channels)
    weighted_input = Multiply()([input_layer, reshape_weights])
    
    # Add the outputs from both paths
    combined_output = Add()([max_pool, weighted_input])
    
    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Two additional fully connected layers for classification
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model