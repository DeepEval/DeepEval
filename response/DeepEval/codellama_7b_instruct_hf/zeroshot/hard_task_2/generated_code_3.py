from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Define the first group
    group1 = Conv2D(16, (1, 1), padding='same', activation='relu')(groups[0])
    group1 = Conv2D(16, (3, 3), padding='same', activation='relu')(group1)
    group1 = Conv2D(16, (1, 1), padding='same', activation='relu')(group1)

    # Define the second group
    group2 = Conv2D(32, (1, 1), padding='same', activation='relu')(groups[1])
    group2 = Conv2D(32, (3, 3), padding='same', activation='relu')(group2)
    group2 = Conv2D(32, (1, 1), padding='same', activation='relu')(group2)

    # Define the third group
    group3 = Conv2D(64, (1, 1), padding='same', activation='relu')(groups[2])
    group3 = Conv2D(64, (3, 3), padding='same', activation='relu')(group3)
    group3 = Conv2D(64, (1, 1), padding='same', activation='relu')(group3)

    # Combine the output of each group
    main_path = Add()([group1, group2, group3])

    # Fuse the main path with the original input layer
    output_layer = Add()([main_path, input_layer])

    # Flatten the output
    output_layer = Flatten()(output_layer)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model