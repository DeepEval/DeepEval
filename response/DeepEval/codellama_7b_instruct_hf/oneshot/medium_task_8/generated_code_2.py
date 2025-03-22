import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    main_path = input_layer
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_path)
    # Process the second group with a 3x3 convolution
    conv_layer = Conv2D(32, (3, 3), activation='relu')(split_layer[1])
    # Combine the second and third groups
    combined_layer = Concatenate()([split_layer[1], conv_layer])
    # Pass through an additional 3x3 convolution
    conv_layer = Conv2D(32, (3, 3), activation='relu')(combined_layer)
    # Concatenate the outputs of all three groups
    output_layer = Concatenate()([main_path, branch_path, conv_layer])
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model