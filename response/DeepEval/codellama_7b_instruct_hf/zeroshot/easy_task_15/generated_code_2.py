from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the specialized block
    specialized_block = Conv2D(32, (3, 3), activation='relu')(input_shape)
    specialized_block = Conv2D(32, (1, 1), activation='relu')(specialized_block)
    specialized_block = Conv2D(32, (1, 1), activation='relu')(specialized_block)
    specialized_block = MaxPooling2D((2, 2))(specialized_block)
    specialized_block = Dropout(0.25)(specialized_block)

    # Define the global average pooling layer
    global_average_pooling = Conv2D(1, (1, 1), activation='relu')(specialized_block)
    global_average_pooling = MaxPooling2D((2, 2))(global_average_pooling)

    # Define the flattening layer
    flatten = Flatten()(global_average_pooling)

    # Define the fully connected layer
    fc_layer = Dense(10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc_layer)

    return model