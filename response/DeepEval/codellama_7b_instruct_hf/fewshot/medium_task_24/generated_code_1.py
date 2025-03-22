import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first branch
    conv1_1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    conv1_2 = Conv2D(64, (3, 3), activation='relu')(conv1_1)
    dropout_1 = Dropout(0.2)(conv1_2)

    # Define the second branch
    conv2_1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    conv2_2 = Conv2D(64, (1, 7), activation='relu')(conv2_1)
    conv2_3 = Conv2D(64, (7, 1), activation='relu')(conv2_2)
    conv2_4 = Conv2D(32, (3, 3), activation='relu')(conv2_3)
    dropout_2 = Dropout(0.2)(conv2_4)

    # Define the third branch
    max_pooling = MaxPooling2D((2, 2))(input_layer)
    dropout_3 = Dropout(0.2)(max_pooling)

    # Concatenate the outputs of all branches
    outputs = keras.layers.concatenate([dropout_1, dropout_2, dropout_3])

    # Flatten the output and add a fully connected layer
    flatten = Flatten()(outputs)
    fc_1 = Dense(128, activation='relu')(flatten)
    fc_2 = Dense(64, activation='relu')(fc_1)
    output_layer = Dense(10, activation='softmax')(fc_2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model