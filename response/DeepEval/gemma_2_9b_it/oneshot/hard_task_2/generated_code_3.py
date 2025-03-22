import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Process each group
    group1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group1)
    group1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1)

    group2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    group2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group2)

    group3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group3)
    group3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group3)

    # Combine the processed groups
    main_path = Add()([group1, group2, group3])

    # Fuse with the original input
    output = Add()([main_path, input_layer])

    # Flatten and classify
    flatten_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model