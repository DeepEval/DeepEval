import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))

    # Compress input channels with 1x1 convolutional layer
    conv_compress = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Expand features through parallel convolutional layers
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_compress)
    conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_compress)
    
    # Concatenate results of parallel convolutional layers
    concat_features = Concatenate()([conv_1x1, conv_3x3])

    # Apply batch normalization
    batch_norm = BatchNormalization()(concat_features)

    # Flatten feature map
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()