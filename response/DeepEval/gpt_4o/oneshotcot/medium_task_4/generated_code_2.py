import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path1: Two blocks of convolution followed by average pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_2)
    
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2_2)
    
    # Path2: Single convolution layer
    conv_single = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine Path1 and Path2
    combined_features = Add()([avg_pool2, conv_single])
    
    # Flatten the combined features
    flatten_layer = Flatten()(combined_features)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model