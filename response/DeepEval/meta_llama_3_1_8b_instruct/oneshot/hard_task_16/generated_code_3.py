import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras import backend as K
from tensorflow.keras import regularizers

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input along the last dimension
        split_tensor = Lambda(lambda x: K.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Three groups of convolutions
        conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[:, :, :, 0])
        conv1_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[:, :, :, 1])
        conv1_3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
        
        # Concatenate the outputs from the three groups
        output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3])
        
        return output_tensor
    
    # Block 1 output
    block1_output = block1(input_layer)
    
    # Transition convolution layer
    transition_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)
    
    # Block 2
    def block2(input_tensor):
        # Global max pooling
        global_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        
        # Two fully connected layers for generating weights
        weights = Flatten()(global_max_pool)
        weights = Dense(units=16, activation='relu')(weights)
        weights = Dense(units=16, activation='relu')(weights)
        
        # Reshape the weights
        weights = Reshape(target_shape=(1, 1, 16))(weights)
        
        # Multiply the weights with the adjusted output
        output_tensor = Multiply()([weights, transition_conv])
        
        return output_tensor
    
    # Block 2 output
    block2_output = block2(transition_conv)
    
    # Main path output
    main_path_output = block2_output
    
    # Branch output
    branch_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the main path and branch outputs
    output = Add()([main_path_output, branch_output])
    
    # Final classification layer
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

def main():
    model = dl_model()
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

if __name__ == "__main__":
    main()