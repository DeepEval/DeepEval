import keras
from keras.layers import Input, Lambda, Conv2D, Dropout, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input channels
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Process each channel group
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Dropout(0.2)(group1)

    group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    group2 = Dropout(0.2)(group2)

    group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    group3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
    group3 = Dropout(0.2)(group3)

    # Concatenate processed groups
    main_path = Concatenate()([group1, group2, group3])

    # Parallel branch pathway
    branch_path = Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine pathways
    combined_path = Add()([main_path, branch_path])

    # Flatten and classify
    flatten_layer = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model