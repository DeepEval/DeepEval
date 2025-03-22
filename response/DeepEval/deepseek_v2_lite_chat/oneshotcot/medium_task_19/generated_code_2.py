import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch
    first_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    first_branch = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(first_branch)
    
    # Second branch
    second_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_branch)
    second_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_branch)
    second_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(second_branch)
    
    # Third branch
    third_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_branch)
    third_branch = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(third_branch)
    third_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(third_branch)
    
    # Fourth branch
    fourth_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_branch)
    fourth_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(fourth_branch)
    fourth_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fourth_branch)
    
    # Concatenate all branches
    concatenated = Concatenate()(list(map(lambda branch: branch, [first_branch, second_branch, third_branch, fourth_branch])))
    
    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Flatten layer
    flat = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model