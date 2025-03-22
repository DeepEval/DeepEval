import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block for max pooling and preprocessing
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(block1)
    block1 = Flatten()(block1)
    block1 = BatchNormalization()(block1)
    
    # Dropout to mitigate overfitting
    block1 = keras.layers.Dropout(rate=0.5)(block1)
    
    # Split block1 into four groups using Lambda layer
    split_block1 = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=2))(block1)
    
    # Second block for feature extraction
    separable_conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_block1[0])
    separable_conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_block1[1])
    separable_conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_block1[2])
    separable_conv4 = Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(split_block1[3])
    
    # Concatenate the outputs from the separable convolutions
    concat_layer = Concatenate(axis=-1)([separable_conv1, separable_conv2, separable_conv3, separable_conv4])
    
    # Reshape the concatenated output
    reshape_layer = Reshape((-1, 1))(concat_layer)
    
    # Output layer
    dense1 = Dense(units=128, activation='relu')(reshape_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model