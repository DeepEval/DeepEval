import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    # 1x1 average pooling layer with stride 1x1
    pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    # 2x2 average pooling layer with stride 2x2
    pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(pool_1)
    # 4x4 average pooling layer with stride 4x4
    pool_3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(pool_2)
    # Flatten the output of the pooling layers
    flattened_output = Flatten()(pool_3)
    
    # Second block
    # 1x1 convolution layer with stride 1x1
    conv_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(flattened_output)
    # 3x3 convolution layer with stride 1x1
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv_1)
    # 3x3 convolution layer with stride 1x1
    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv_2)
    # 1x1 convolution layer with stride 1x1
    conv_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv_3)
    # Dropout layer with rate 0.5
    dropout_1 = Dropout(rate=0.5)(conv_4)
    # Dropout layer with rate 0.5
    dropout_2 = Dropout(rate=0.5)(dropout_1)
    
    # Concatenate the outputs of the parallel paths
    concatenated_output = Concatenate()([dropout_1, dropout_2])
    
    # Fully connected layers
    dense_1 = Dense(units=128, activation='relu')(concatenated_output)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model