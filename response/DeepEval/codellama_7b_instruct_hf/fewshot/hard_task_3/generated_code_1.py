import keras
from keras.layers import Input, Lambda, Conv2D, Dropout, Concatenate, Add, Flatten, Dense

def dl_model():
    # Split the input into three groups along the channel dimension
    input_layer = Input(shape=(32, 32, 3))
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Sequence of 1x1 convolution and 3x3 convolution for each group
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)

    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Dropout(rate=0.2)(branch_path)

    # Combine the outputs from both pathways
    adding_layer = Add()([main_path, branch_path])

    # Flatten and pass through a fully connected layer
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model