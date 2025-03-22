import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dropout, Add, Flatten, Dense

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channels groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)

    # Process each channel group
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
    group1 = Dropout(0.2)(group1)

    group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    group2 = Dropout(0.2)(group2)

    group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    group3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
    group3 = Dropout(0.2)(group3)

    # Concatenate the outputs from each channel group
    concat_layer = Concatenate()( [group1, group2, group3] )

    # Branch pathway
    branch_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the main and branch pathways
    combined_output = Add()([concat_layer, branch_output])

    # Flatten and fully connect
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model