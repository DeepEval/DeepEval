import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting the input into three channels and applying convolutions
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_inputs[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split_inputs[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split_inputs[2])
    
    # Applying dropout
    dropout = Dropout(0.5)(Concatenate()([conv1, conv2, conv3]))

    # Block 2: Four branches for feature extraction
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(dropout)
    
    path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(dropout)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path2)
    
    path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(dropout)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(path3)
    
    path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(dropout)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path4)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout)
    
    # Concatenating outputs from all paths
    concatenated_output = Concatenate()([path1, path2, path3, path4])

    # Flattening and fully connected layers
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model