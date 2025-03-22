import keras
from keras.layers import Input, Reshape, Lambda, Permute, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    input_shape = input_layer.shape

    # Reshape the input tensor into three groups
    reshaped = Reshape(target_shape=(input_shape[0], input_shape[1], 3, input_shape[3] // 3))(input_layer)

    # Swap the third and fourth dimensions using a permutation operation
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape the tensor back to its original input shape
    reshaped_back = Reshape(target_shape=input_shape)(permuted)

    # Pass the output through a fully connected layer with a softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(reshaped_back)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model