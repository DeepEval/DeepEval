import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer) 
    
    # Group 1
    conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    
    # Group 2
    conv2_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    
    # Group 3
    conv3_1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Concatenate outputs of groups
    main_path_output = Concatenate(axis=2)([conv1_1, conv2_1, conv3_1])

    # Branch Path
    branch_path = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Add outputs
    output = keras.layers.add([main_path_output, branch_path])

    # Flatten and fully connected layers
    x = Flatten()(output)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model