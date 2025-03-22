import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Split the input channels
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)

    # Process each channel group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Dropout(0.2)(group1)

    group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    group2 = Dropout(0.2)(group2)

    group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
    group3 = Dropout(0.2)(group3)

    # Concatenate the outputs from the channel groups
    main_pathway = Concatenate(axis=2)([group1, group2, group3])

    # Branch pathway
    branch_pathway = Conv2D(filters=64 * 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the pathways
    combined_output = tf.keras.layers.Add()([main_pathway, branch_pathway])

    # Flatten and classify
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model