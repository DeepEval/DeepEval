import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.regularizers import l2

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add 1x1 initial convolutional layer
    conv_initial = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', 
                          kernel_regularizer=l2(0.01))(input_layer)
    
    # Step 3: Define main path
    def main_path(input_tensor):
        # Branch 1: 3x3 convolutional layer
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                         kernel_regularizer=l2(0.01))(input_tensor)
        
        # Branch 2: Downsample using max pooling, then 3x3 convolutional layer, and finally upsample using transpose convolution
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', 
                         kernel_regularizer=l2(0.01))(input_tensor)
        branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu', 
                                 kernel_regularizer=l2(0.01))(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                         kernel_regularizer=l2(0.01))(branch2)
        
        # Branch 3: Downsample using max pooling, then 3x3 convolutional layer, and finally upsample using transpose convolution
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', 
                         kernel_regularizer=l2(0.01))(input_tensor)
        branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu', 
                                 kernel_regularizer=l2(0.01))(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                         kernel_regularizer=l2(0.01))(branch3)
        
        # Concatenate outputs from all branches
        output_tensor = Concatenate()([conv_initial, branch1, branch2, branch3])
        
        return output_tensor
    
    # Step 4: Add 1x1 convolutional layer after concatenation
    output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', 
                           kernel_regularizer=l2(0.01))(output_tensor)
    
    # Step 5: Define branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', 
                         kernel_regularizer=l2(0.01))(output_tensor)
    
    # Step 6: Add main path and branch path
    output_tensor = Add()([output_tensor, branch_path])
    
    # Step 7: Flatten and add dense layer
    flatten_layer = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add another dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Call the dl_model function to get the constructed model
model = dl_model()