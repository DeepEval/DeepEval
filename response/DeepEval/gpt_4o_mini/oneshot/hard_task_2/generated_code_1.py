import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def process_group(input_tensor):
        # Series of convolutions: 1x1 -> 3x3 -> 1x1
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each group through the defined function
    processed_groups = [process_group(group) for group in split_tensors]

    # Combine the outputs from the three groups using addition
    main_path = Add()(processed_groups)

    # Fuse the main path with the original input
    fused_output = Add()([main_path, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(fused_output)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model