from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Concatenate
from keras.applications.vgg16 import VGG16



def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the 1x1 convolution branch
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the 1x1 convolution + 3x3 convolution branch
    conv2 = Conv2D(32, (1, 1), activation='relu')(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv2)

    # Define the 1x1 convolution + 5x5 convolution branch
    conv3 = Conv2D(32, (1, 1), activation='relu')(conv1)
    conv3 = Conv2D(32, (5, 5), activation='relu')(conv3)

    # Define the 3x3 max pooling + 1x1 convolution branch
    pool = MaxPooling2D((3, 3))(conv1)
    conv4 = Conv2D(32, (1, 1), activation='relu')(pool)

    # Define the concatenation layer
    concat = Concatenate()([conv2, conv3, conv4])

    # Define the flatten layer
    flatten = Flatten()(concat)

    # Define the fully connected layers
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the summary of the model
    print(model.summary())

    return model