import keras
from keras.layers import Input, Reshape, Lambda, Permute, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Reshape the input tensor into three groups
    reshape = Reshape(target_shape=(32, 32, 3, 3))(input_layer)

    # Swap the third and fourth dimensions for channel shuffling
    permute = Permute((1, 2, 4, 3))(reshape)

    # Reshape the tensor back to its original input shape
    reshaped = Reshape(target_shape=(32, 32, 3))(permute)

    # Pass the output through a fully connected layer with a softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(reshaped)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model