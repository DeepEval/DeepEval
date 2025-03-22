import keras
from keras.layers import Input, Conv2D, Lambda, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups
    split_layer = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each group
    group1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_layer[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group1)
    group1 = Dropout(0.2)(group1)

    group2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_layer[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    group2 = Dropout(0.2)(group2)

    group3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_layer[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group3)
    group3 = Dropout(0.2)(group3)

    # Concatenate outputs of groups
    main_path = Concatenate()([group1, group2, group3])

    # Branch pathway
    branch_path = Conv2D(filters=192, kernel_size=(1, 1), activation='relu')(input_layer)

    # Combine pathways
    combined_path = Add()([main_path, branch_path])

    # Flatten and classify
    flatten_layer = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model