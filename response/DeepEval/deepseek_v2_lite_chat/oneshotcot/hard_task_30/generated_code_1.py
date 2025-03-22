import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Dual-path structure
    def first_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)  # Restore channels
        add = Add()([input_tensor, conv2])
        
        return add
    
    block1_output = first_block(input_tensor=input_layer)
    batch_norm1 = BatchNormalization()(block1_output)
    flatten_layer1 = Flatten()(batch_norm1)
    
    # Second block: Split input into three groups
    def second_block(input_tensor):
        split = Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)
        
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        
        concat = Concatenate()(pooling_outputs)  # Concatenate after pooling
        
        return concat
    
    block2_output = second_block(input_tensor=flatten_layer1)
    batch_norm2 = BatchNormalization()(block2_output)
    dense1 = Dense(units=128, activation='relu')(batch_norm2)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model