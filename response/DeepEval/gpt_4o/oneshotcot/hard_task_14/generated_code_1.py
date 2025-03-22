import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape, Flatten

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense_main1 = Dense(units=16, activation='relu')(global_avg_pooling)  # Arbitrarily chosen units for illustration
    dense_main2 = Dense(units=3, activation='sigmoid')(dense_main1)  # Output units match number of input channels (3)
    weights = Reshape((1, 1, 3))(dense_main2)  # Reshape to match input dimensions
    weighted_main = Multiply()([input_layer, weights])  # Element-wise multiplication with input feature map
    
    # Branch path
    conv_branch = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine paths
    combined = Add()([weighted_main, conv_branch])
    
    # Fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # Output units for CIFAR-10 classification

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model