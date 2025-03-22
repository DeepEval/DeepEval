import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer) 
    
    # Group 1 (1x1 conv)
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])

    # Group 2 (3x3 conv)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x[1])

    # Group 3 (5x5 conv)
    conv3_3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x[2])

    # Concatenate outputs from the groups
    main_path = Concatenate(axis=1)([conv1_1, conv2_2, conv3_3])

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)

    # Fuse features
    fused_features = main_path + branch_path

    # Classification Layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model