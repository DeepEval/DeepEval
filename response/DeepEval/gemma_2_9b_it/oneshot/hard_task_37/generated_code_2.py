import keras
from keras.layers import Input, Conv2D, concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Define the block
    def block(input_tensor):
      conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
      conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
      conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

      # Separate paths
      path1 = conv1
      path2 = conv2
      path3 = conv3

      # Parallel branch
      parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

      # Combine paths
      combined = keras.layers.add([path1, path2, path3, parallel_branch])

      return combined

    # Build the model
    branch1 = block(input_layer)
    branch2 = block(input_layer)

    # Concatenate branches
    concat_layer = concatenate([branch1, branch2])

    # Flatten and fully connected layers
    flatten = Flatten()(concat_layer)
    dense_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model