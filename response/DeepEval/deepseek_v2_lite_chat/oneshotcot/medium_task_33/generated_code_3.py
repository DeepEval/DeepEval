import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_1 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    split_2 = Lambda(lambda x: keras.backend.split(x[0], num_or_size_splits=3, axis=-1))([input_layer, split_1[2]])
    split_3 = Lambda(lambda x: keras.backend.split(x[0], num_or_size_splits=3, axis=-1))([input_layer, split_1[1]])
    
    # Separate convolutional layers for each channel group
    conv_1x1_1 = Conv2D(32, (1, 1), padding='same')(split_1[0])
    conv_1x1_2 = Conv2D(32, (1, 1), padding='same')(split_2[0])
    conv_1x1_3 = Conv2D(32, (1, 1), padding='same')(split_3[0])
    
    conv_3x3_1 = Conv2D(64, (3, 3), padding='same')(split_1[1])
    conv_3x3_2 = Conv2D(64, (3, 3), padding='same')(split_2[1])
    conv_3x3_3 = Conv2D(64, (3, 3), padding='same')(split_3[1])
    
    conv_5x5_1 = Conv2D(64, (5, 5), padding='same')(split_1[2])
    conv_5x5_2 = Conv2D(64, (5, 5), padding='same')(split_2[2])
    conv_5x5_3 = Conv2D(64, (5, 5), padding='same')(split_3[2])
    
    # Concatenate the outputs from the three groups
    concat = Concatenate()(
        [conv_1x1_1, conv_3x3_1, conv_5x5_1, conv_1x1_2, conv_3x3_2, conv_5x5_2, conv_1x1_3, conv_3x3_3, conv_5x5_3])
    
    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()