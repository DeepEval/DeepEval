import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, Reshape
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block processing
    def first_block(x):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(x)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(x)
        
        # Flatten each pooled output
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)
        
        # Apply dropout to reduce overfitting
        combined = Dropout(0.5)(Concatenate()([flat1, flat2, flat3]))
        
        # Reshape for second block
        reshaped = Reshape((3, 3, 4))(combined)
        return reshaped
    
    reshaped_output = first_block(input_layer)
    
    # Second block processing
    def second_block(x):
        # Split the input into four groups
        split_1 = Lambda(lambda z: tf.split(z, 1, axis=-1))(x)
        split_2 = Lambda(lambda z: tf.split(z, 1, axis=-1))(x)
        split_3 = Lambda(lambda z: tf.split(z, 1, axis=-1))(x)
        split_4 = Lambda(lambda z: tf.split(z, 1, axis=-1))(x)
        
        # Process each group with separable convolutional layers
        conv1 = Conv2D(32, (1, 1), activation='relu')(split_1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(split_2)
        conv3 = Conv2D(32, (5, 5), activation='relu', padding='same')(split_3)
        conv4 = Conv2D(32, (7, 7), activation='relu', padding='same')(split_4)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1, conv2, conv3, conv4])
        return concatenated
    
    second_block_output = second_block(reshaped_output)
    
    # Flatten the output and pass through fully connected layers
    flattened = Flatten()(second_block_output)
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()