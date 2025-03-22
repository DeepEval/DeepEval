import keras
from keras.layers import Input, Conv2D, Lambda, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=2))(input_layer)

    # Process each group
    group1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group1)
    group1 = Dropout(0.2)(group1)
    
    group2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    group2 = Dropout(0.2)(group2)

    group3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensor[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group3)
    group3 = Dropout(0.2)(group3)

    # Concatenate outputs from groups
    main_path = Concatenate(axis=2)([group1, group2, group3])

    # Parallel branch pathway
    branch_path = Conv2D(filters=192, kernel_size=(1, 1), activation='relu')(input_layer)

    # Combine pathways
    output = Add()([main_path, branch_path])

    # Flatten and classify
    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model