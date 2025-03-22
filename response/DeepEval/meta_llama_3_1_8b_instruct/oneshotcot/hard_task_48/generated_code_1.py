import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Lambda, Reshape
import tensorflow as tf
from tensorflow.keras import layers

def dl_model():     
    # Step 1: add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: define block 1
    def block1(input_tensor):
        # Split the input into three groups
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Define the separable convolutional layers for each group
        path1 = layers.SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(split_tensor[0])
        path2 = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(split_tensor[1])
        path3 = layers.SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same')(split_tensor[2])
        
        # Concatenate the outputs of the three groups
        output_tensor = Concatenate()([path1, path2, path3])
        
        # Apply batch normalization
        output_tensor = BatchNormalization()(output_tensor)
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Step 3: define block 2
    def block2(input_tensor):
        # Define the four parallel paths
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Reshape((32, 32, 1))(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Reshape((32, 32, 64))(path3)
        
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Reshape((32, 32, 64))(path4)
        
        # Concatenate the outputs of the four paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        # Apply batch normalization
        output_tensor = BatchNormalization()(output_tensor)
        
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Step 4: apply batch normalization, flatten, and dense layers
    bath_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Step 5: build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model