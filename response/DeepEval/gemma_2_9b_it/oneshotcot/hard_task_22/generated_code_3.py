import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)
    
    # Multi-scale Feature Extraction
    x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x[0])
    x1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x1)
    x1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x1)

    x2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x[1])
    x2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x2)
    x2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x2)

    x3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x[2])
    x3 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x3)
    x3 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x3)

    # Concatenate Outputs
    main_path_output = Concatenate(axis=2)([x1, x2, x3])

    # Branch Path
    branch_path_output = Conv2D(filters=48, kernel_size=(1, 1), activation='relu')(input_layer)

    # Fusion
    fused_output = main_path_output + branch_path_output

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model