import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Basic block
    def basic_block(input_tensor, filters):
        conv = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn

    # Main structure block
    def main_block(input_tensor, filters):
        block1 = basic_block(input_tensor, filters)
        block2 = basic_block(block1, filters)
        return block2

    # Branch for feature extraction
    def branch(input_tensor):
        conv = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool = AveragePooling2D(pool_size=(1, 1))(conv)
        return avg_pool

    # Combine outputs
    branch_output = branch(input_tensor=input_layer)
    main_output = main_block(input_tensor=input_layer, filters=64)
    combined_output = Concatenate()([branch_output, main_output])

    # Additional layers
    bn = BatchNormalization()(combined_output)
    flatten = Flatten()(bn)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the model
model = dl_model()
model.summary()