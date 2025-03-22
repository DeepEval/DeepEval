import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate the output of conv_1 with the input_layer
    concat_1 = Concatenate(axis=3)([conv_1, input_layer])
    
    # Second convolutional layer
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_1)
    
    # Concatenate the output of conv_2 with the output of conv_1
    concat_2 = Concatenate(axis=3)([conv_2, conv_1])
    
    # Third convolutional layer
    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_2)
    
    # Concatenate the output of conv_3 with the output of conv_2
    concat_3 = Concatenate(axis=3)([conv_3, conv_2])
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat_3)
    
    # Flatten layer
    flatten_layer = Flatten()(max_pooling)
    
    # First fully connected layer
    dense_1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    dense_2 = Dense(units=10, activation='softmax')(dense_1)
    
    # Model definition
    model = keras.Model(inputs=input_layer, outputs=dense_2)
    
    return model