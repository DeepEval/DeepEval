import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    # Block for the first branch
    block1_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block1_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1_output)
    
    # Branch 2
    # Block for the second branch
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(block2_output)
    
    # Concatenate the outputs of the two branches
    concat_output = Concatenate()([block1_output, block2_output])
    
    # Flatten and pass through three dense layers
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.summary()