import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dropout, Add
from keras.layers.experimental import preprocessing
from tensorflow import tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channels
    split_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Main pathway
    output_branches = []
    for branch_input in split_input:
      conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(branch_input)
      conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
      dropout = Dropout(0.2)(conv2)
      output_branches.append(dropout)

    # Concatenate outputs from the three branches
    main_output = Concatenate()(output_branches)

    # Branch pathway
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Combine main and branch outputs
    combined_output = Add()([main_output, branch_output])

    # Fully connected layer
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model