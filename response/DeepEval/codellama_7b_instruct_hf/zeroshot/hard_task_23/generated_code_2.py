from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the initial 1x1 convolutional layer
    init_conv = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the first branch of the model
    branch1 = Conv2D(32, (3, 3), activation='relu')(init_conv)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)

    # Define the second branch of the model
    branch2 = MaxPooling2D((2, 2))(init_conv)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)

    # Define the third branch of the model
    branch3 = MaxPooling2D((2, 2))(init_conv)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Define the concatenate layer
    concat = concatenate([branch1, branch2, branch3], axis=1)

    # Define the refinement layer
    refined = Conv2D(32, (1, 1), activation='relu')(concat)

    # Define the fully connected layer
    fc = Flatten()(refined)
    fc = Dense(10, activation='softmax')(fc)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model