import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense, tf

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
      split_tensor = Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)
      conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
      conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
      conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
      bn1 = BatchNormalization()(conv1)
      bn2 = BatchNormalization()(conv2)
      bn3 = BatchNormalization()(conv3)
      return Concatenate()([bn1, bn2, bn3])

    block1_output = block1(input_layer)

    # Block 2
    def block2(input_tensor):
      path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      path2 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)
      path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
      path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      path3_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
      path3_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
      path3 = Concatenate()([path3_1, path3_2])
      path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      path4_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
      path4_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
      path4_3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
      path4 = Concatenate()([path4_1, path4_2, path4_3])
      return Concatenate()([path1, path2, path3, path4])
    
    block2_output = block2(block1_output)
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model