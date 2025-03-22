import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Split the input into three groups using Lambda layer
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 3: Define the main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])
    
    # Step 4: Concatenate the outputs of the main path
    main_path_output = Concatenate()([conv1, conv2, conv3])
    
    # Step 5: Define the branch path
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 6: Combine the outputs of the main and branch paths through addition
    fused_output = Add()([main_path_output, branch_output])
    
    # Step 7: Flatten the output for the dense layers
    flatten_layer = Flatten()(fused_output)

    # Step 8: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 9: Add second dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model