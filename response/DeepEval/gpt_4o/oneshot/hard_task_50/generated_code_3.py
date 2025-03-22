import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, SeparableConv2D, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    dropout1 = Dropout(rate=0.5)(flat1)
    dropout2 = Dropout(rate=0.5)(flat2)
    dropout3 = Dropout(rate=0.5)(flat3)
    
    concat1 = Concatenate()([dropout1, dropout2, dropout3])
    
    dense1 = Dense(units=256, activation='relu')(concat1)
    
    reshape1 = Reshape(target_shape=(8, 8, 4))(dense1)  # Reshaping to 4D tensor

    # Second block
    def split_and_process(input_tensor):
        splits = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(splits[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(splits[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(splits[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same')(splits[3])
        return Concatenate()([conv1, conv2, conv3, conv4])
    
    processed_tensor = Lambda(split_and_process)(reshape1)
    
    flatten2 = Flatten()(processed_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model