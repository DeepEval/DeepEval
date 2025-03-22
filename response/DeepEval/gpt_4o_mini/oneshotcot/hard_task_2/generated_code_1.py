import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Function to create the convolutional path
    def create_path(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Step 2: Process each split through the convolutional path
    path1 = create_path(split[0])
    path2 = create_path(split[1])
    path3 = create_path(split[2])

    # Step 3: Combine the outputs from the three paths using addition
    combined_path = Add()([path1, path2, path3])

    # Step 4: Fuse the combined path with the original input layer
    fused_output = Add()([combined_path, input_layer])

    # Step 5: Flatten the combined features
    flatten_layer = Flatten()(fused_output)

    # Step 6: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model