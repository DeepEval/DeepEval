import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model
import tensorflow as tf

# Define the main path and branch path for the first block
def main_branch(input_tensor):
    conv = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    conv = Conv2D(32, (3, 3), padding='same', activation='relu')(conv)
    dropout = keras.layers.Dropout(0.5)(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(dropout)
    return pool

def branch(input_tensor):
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    paths = []
    for path_tensor in split:
        conv = Conv2D(64, (1, 1), activation='relu')(path_tensor)
        conv = Conv2D(64, (3, 3), activation='relu')(conv)
        conv = Conv2D(64, (5, 5), activation='relu')(conv)
        paths.append(conv)
    branch_output = Concatenate()(paths)
    return branch_output

def second_block(input_tensor):
    conv = Conv2D(64, (1, 1), activation='relu')(input_tensor)
    conv = Conv2D(64, (3, 3), activation='relu')(conv)
    conv = Conv2D(64, (5, 5), activation='relu')(conv)
    dropout = keras.layers.Dropout(0.5)(conv)
    return dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    first_block_output = main_branch(input_layer)
    first_block_output = batch_norm(first_block_output)
    first_block_output = Flatten()(first_block_output)
    
    branch_output = branch(input_layer)
    second_block_output = second_block(branch_output)
    
    # Concatenate and pass through fully connected layers
    concatenated = Concatenate()([first_block_output, second_block_output])
    dense = Dense(256, activation='relu')(concatenated)
    output = Dense(10, activation='softmax')(dense)
    
    # Model architecture
    model = Model(inputs=input_layer, outputs=output)
    
    return model

def batch_norm(x):
    return BatchNormalization()(x)

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have a loaded and preprocessed CIFAR-10 dataset
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))