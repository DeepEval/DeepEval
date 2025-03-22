import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, tf

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Group 1
    x1 = x[0]
    
    # Group 2
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    
    # Group 3
    x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Combine Group 2 and 3
    x23 = Concatenate()([x2, x3])
    
    # Main Path Output
    main_path = Concatenate()([x1, x23])

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse Paths
    fused_output = main_path + branch_path

    # Final Classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model