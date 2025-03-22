import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    main_path_1 = x[0]
    main_path_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x[1])
    main_path_3 = x[2]
    
    main_path_4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(Concatenate()([main_path_2, main_path_3]))
    
    main_path = Concatenate()([main_path_1, main_path_4])
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Fusion
    combined_output = Add()([main_path, branch_path])
    
    # Classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model