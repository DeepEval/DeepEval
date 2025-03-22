import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 2: Create three separate paths
    def create_path(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    # Apply the paths to each split
    path_outputs = [create_path(group) for group in split]

    # Step 3: Combine the outputs of the three paths using addition
    main_path = Add()(path_outputs)

    # Step 4: Fuse the main path with the original input layer
    combined = Add()([main_path, input_layer])

    # Step 5: Flatten the combined features
    flatten_layer = Flatten()(combined)

    # Step 6: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model