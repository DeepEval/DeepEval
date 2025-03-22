import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Flatten, Dense, Reshape

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))  

    def block1(input_tensor):
      path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
      output_tensor = Concatenate()([path1, path2, path3, path4])
      return output_tensor

    block1_output = block1(input_layer)
    
    # Block 2
    block2_output = GlobalAveragePooling2D()(block1_output) 
    dense1 = Dense(units=block2_output.shape[-1], activation='relu')(block2_output)  
    dense2 = Dense(units=block2_output.shape[-1], activation='relu')(dense1)
    reshape_layer = Reshape((block1_output.shape[1], block1_output.shape[2], block2_output.shape[-1]))(dense2)
    
    # Element-wise multiplication
    elementwise_product = keras.layers.Multiply()([block1_output, reshape_layer])
    output_layer = Flatten()(elementwise_product)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model