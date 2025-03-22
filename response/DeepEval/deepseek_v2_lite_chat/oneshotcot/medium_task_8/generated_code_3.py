import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # Process the first group unchanged
    first_group = split[0]
    
    # Process the second group with a 3x3 convolutional layer
    second_group = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split[1])
    
    # Process the third group after combining with the second group
    third_group = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tf.concat([split[2], second_group], axis=-1))
    
    # Concatenate the outputs from all three groups
    concat = Concatenate()(list(split) + [third_group])
    
    # Branch path with a 1x1 convolutional layer
    branch = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split[0])
    
    # Batch normalization, flattening, and fully connected layers
    batch_norm = BatchNormalization()(concat)
    flat = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Instantiate and return the model
model = dl_model()