import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Splitting the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def block(input_tensor):
        # Main pathway
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        dropout = Dropout(0.5)(conv2)  # Dropout for feature selection
        return dropout

    # Processing each group
    group1 = block(split_layer[0])
    group2 = block(split_layer[1])
    group3 = block(split_layer[2])

    # Concatenating the outputs from the three groups
    main_path = Add()([group1, group2, group3])

    # Branch pathway
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Adding the main path and branch path
    adding_layer = Add()([main_path, branch_path])

    # Flattening the output
    flatten_layer = Flatten()(adding_layer)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model