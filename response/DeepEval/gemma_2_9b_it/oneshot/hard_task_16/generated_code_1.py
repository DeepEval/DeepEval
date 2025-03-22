import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, Dense, Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
      split_tensor = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
      conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
      conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
      conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
      
      conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
      conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
      conv2_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_2)
      
      conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
      conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
      conv3_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3_2)
      
      output_tensor = Concatenate()([conv1_3, conv2_3, conv3_3])
      
      return output_tensor

    block1_output = block_1(input_layer)
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1_output)

    # Block 2
    def block_2(input_tensor):
      pooled_output = MaxPooling2D(pool_size=(7, 7), strides=1, padding='same')(transition_conv)
      
      dense1 = Dense(units=pooled_output.shape[-1], activation='relu')(Flatten()(pooled_output))
      dense2 = Dense(units=pooled_output.shape[-1], activation='relu')(dense1)
      weights = Reshape((pooled_output.shape[1], pooled_output.shape[2], pooled_output.shape[3]))(dense2)
      output_tensor = tf.multiply(pooled_output, weights)
      
      return output_tensor

    block2_output = block_2(transition_conv)

    # Branch
    branch_output = Input(shape=(32, 32, 3))
    
    # Combine outputs and classification
    combined_output = tf.add(block2_output, branch_output)
    output_layer = Dense(units=10, activation='softmax')(Flatten()(combined_output))

    model = keras.Model(inputs=[input_layer, branch_output], outputs=output_layer)

    return model