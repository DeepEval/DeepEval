from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    branch1 = conv2

    # Define the second branch
    conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    conv4 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    branch2 = conv4

    # Define the third branch
    conv5 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    conv6 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu')
    branch3 = conv6

    # Define the concatenation layer
    concat = Add()([branch1, branch2, branch3])

    # Define the fully connected layer
    flatten = Flatten()(concat)
    dense = Dense(128, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    return model