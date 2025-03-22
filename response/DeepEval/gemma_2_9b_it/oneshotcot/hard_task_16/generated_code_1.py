import keras
from keras.layers import Input, Lambda, Conv2D, concatenate, MaxPooling2D, BatchNormalization, Dense, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
      split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
      
      path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
      path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
      path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)
      
      path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
      path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
      path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

      path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
      path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
      path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path3)
      
      output_tensor = concatenate([path1, path2, path3], axis=-1)
      
      return output_tensor

    block1_output = block1(input_layer)

    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)
    
    block2_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(transition_conv)
    
    block2_dense1 = Dense(units=64 * 64, activation='relu')(block2_output)
    block2_dense2 = Dense(units=64 * 64, activation='relu')(block2_dense1)
    
    block2_weights = Lambda(lambda x: tf.reshape(x, (block2_output.shape[1], block2_output.shape[2], block2_output.shape[3], 64)))(block2_dense2)
    
    block2_output = block2_output * block2_weights
    
    branch_output = input_layer
    
    main_output = block2_output
    final_output = main_output + branch_output
    
    output_layer = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model