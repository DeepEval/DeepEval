import keras
from keras.layers import Input, Conv2D, Lambda, Flatten, Dense, Add, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32 and 3 channels (RGB)

    # Splitting the input into three groups
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = splits[0]
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(splits[1])
    group3 = splits[2]

    # Combining the second group with the third group
    combined_group = Add()([group2, group3])
    main_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)

    # Concatenating the outputs of all three groups
    main_path_final_output = Concatenate()([group1, main_path_output, group3])

    # Branch path processing
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusing the main path and branch path outputs
    fused_output = Add()([main_path_final_output, branch_path_output])

    # Flattening and passing through a fully connected layer
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # To display the model architecture