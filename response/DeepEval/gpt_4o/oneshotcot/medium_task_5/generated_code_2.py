import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2_main = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1_main)
    max_pooling_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_main)
    
    conv3_main = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pooling_main)
    conv4_main = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3_main)
    max_pooling_main_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4_main)
    
    # Branch Path
    conv1_branch = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_branch)
    
    # Combine Main Path and Branch Path
    combined = Add()([max_pooling_main_2, max_pooling_branch])
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model