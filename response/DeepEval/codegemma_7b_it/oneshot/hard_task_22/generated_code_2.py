import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, add

def dl_model(): 
  input_layer = Input(shape=(32, 32, 3))
  
  def main_path(input_tensor):
    split_input = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
    concat_output = Concatenate()([conv1, conv2, conv3])
    return concat_output
  
  def branch_path(input_tensor):
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    return conv
  
  main_output = main_path(input_layer)
  branch_output = branch_path(input_layer)
  
  combined_output = add([main_output, branch_output])
  flatten_output = Flatten()(combined_output)
  dense1 = Dense(units=128, activation='relu')(flatten_output)
  dense2 = Dense(units=64, activation='relu')(dense1)
  output_layer = Dense(units=10, activation='softmax')(dense2)
  
  model = keras.Model(inputs=input_layer, outputs=output_layer)
  
  return model