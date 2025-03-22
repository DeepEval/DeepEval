import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second Convolutional Layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Max Pooling Layer
    pooled_features = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Add pooled features with the input layer
    added_features = Add()([pooled_features, input_layer])
    
    # Flatten the features
    flatten_layer = Flatten()(added_features)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model