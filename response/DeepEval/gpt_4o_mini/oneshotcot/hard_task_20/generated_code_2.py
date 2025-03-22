import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():

    # Step 1: Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Main path processing
    # Splitting the input into three groups
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction with convolutional layers of different kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate outputs from the main path
    main_path_output = Concatenate()([path1, path2, path3])

    # Step 3: Branch path processing
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 4: Combine main and branch paths through addition
    fused_features = Add()([main_path_output, branch_path])

    # Step 5: Flatten and fully connected layers for classification
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model