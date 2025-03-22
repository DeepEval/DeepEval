import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    # Group 1
    main_path_group1 = x[0] 

    # Group 2
    main_path_group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    
    # Group 3
    main_path_group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[2])

    # Combine Groups 2 & 3
    main_path_combined = Concatenate(axis=3)([main_path_group2, main_path_group3])

    # Main Path Output
    main_path_output = main_path_combined 

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Paths
    combined_output = keras.layers.add([main_path_output, branch_path])

    # Flatten and Classify
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model