import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add first convolutional block: Conv2D -> MaxPooling2D
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Step 3: Add second convolutional block: Conv2D -> Conv2D -> MaxPooling2D
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Step 4: Add third convolutional block: Conv2D -> Conv2D -> Conv2D -> MaxPooling2D
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv6)
    
    # Step 5: Flatten the feature maps
    flatten_layer = Flatten()(maxpool3)
    
    # Step 6: Add three fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 7: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model