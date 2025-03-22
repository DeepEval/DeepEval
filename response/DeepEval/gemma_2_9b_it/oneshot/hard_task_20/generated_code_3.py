import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    # Group 1 (1x1 conv)
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x[0]) 
    
    # Group 2 (3x3 conv)
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x[1])
    
    # Group 3 (5x5 conv)
    conv3_1 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x[2])

    # Concatenate outputs of the three groups
    main_path = Concatenate(axis=3)([conv1_1, conv2_1, conv3_1])

    # Branch Path
    branch_conv = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main and branch paths
    fused_features = main_path + branch_conv

    # Classification layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model