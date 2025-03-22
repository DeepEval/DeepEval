import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1
    conv1_path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_path1)
    avg_pooling_path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_path1)
    
    # Path 2
    conv_path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the paths
    combined = Add()([avg_pooling_path1, conv_path2])
    
    # Flatten the combined output
    flatten_layer = Flatten()(combined)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model