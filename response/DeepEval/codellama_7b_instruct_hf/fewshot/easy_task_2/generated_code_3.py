import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the first sequential feature extraction layer
    conv_layer_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_shape)
    avg_pool_layer_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer_1)

    # Define the second sequential feature extraction layer
    conv_layer_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(avg_pool_layer_1)
    avg_pool_layer_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer_2)

    # Define the third sequential feature extraction layer
    conv_layer_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(avg_pool_layer_2)
    avg_pool_layer_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer_3)

    # Flatten the feature maps
    flatten_layer = Flatten()(avg_pool_layer_3)

    # Define the fully connected layers
    dense_layer_1 = Dense(units=512, activation='relu')(flatten_layer)
    dropout_layer_1 = Dropout(rate=0.5)(dense_layer_1)
    dense_layer_2 = Dense(units=256, activation='relu')(dropout_layer_1)
    dropout_layer_2 = Dropout(rate=0.5)(dense_layer_2)

    # Define the output layer
    output_layer = Dense(units=1000, activation='softmax')(dropout_layer_2)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    
    return model