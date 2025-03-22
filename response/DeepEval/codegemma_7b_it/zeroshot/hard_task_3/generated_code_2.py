import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import concatenate

def dl_model():

  inputs = Input(shape=(32, 32, 3))
  splited_inputs = Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)

  main_outputs = []
  for split_input in splited_inputs:
    x = Conv2D(filters=64, kernel_size=(1, 1), padding='valid', activation='relu')(split_input)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    main_outputs.append(x)

  main_output = concatenate(main_outputs)

  branch_output = Conv2D(filters=128, kernel_size=(1, 1), padding='valid', activation='relu')(inputs)
  branch_output = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch_output)
  branch_output = Dropout(rate=0.2)(branch_output)

  output = tf.keras.layers.add([main_output, branch_output])
  output = Flatten()(output)
  output = Dense(units=10, activation='softmax')(output)

  model = Model(inputs=inputs, outputs=output)
  return model