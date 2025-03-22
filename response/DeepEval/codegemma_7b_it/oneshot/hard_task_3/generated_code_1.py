import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Process each group with a sequence of 1x1 convolution, 3x3 convolution, and dropout
    main_pathway = []
    for group in split_layer:
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout = Dropout(rate=0.2)(conv2)
        main_pathway.append(dropout)

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(main_pathway)

    # Process the input through a 1x1 convolution to match the output dimension of the main pathway
    branch_pathway = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs from both pathways using addition
    combined_pathway = keras.layers.add([main_pathway, branch_pathway])

    # Flatten the combined pathway and pass it through three fully connected layers
    flatten_layer = Flatten()(combined_pathway)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model