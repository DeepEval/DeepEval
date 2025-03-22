from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten, Concatenate
from keras.models import Model


def dl_model():
    
    # Define the input shape and the number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # Define the feature extraction paths
    path1 = VGG16(include_top=False, input_shape=input_shape, pooling='avg')
    path2 = VGG16(include_top=False, input_shape=input_shape, pooling='avg')

    # Add 1x1 convolutions to the feature extraction paths
    path1 = path1.add(Conv2D(64, (1, 1), activation='relu'))
    path2 = path2.add(Conv2D(64, (1, 1), activation='relu'))

    # Add a sequence of convolutions to the feature extraction paths
    path1 = path1.add(Conv2D(64, (1, 7), activation='relu'))
    path1 = path1.add(Conv2D(64, (7, 1), activation='relu'))
    path2 = path2.add(Conv2D(64, (1, 7), activation='relu'))
    path2 = path2.add(Conv2D(64, (7, 1), activation='relu'))

    # Concatenate the outputs from the feature extraction paths
    concat = Concatenate()([path1, path2])

    # Add a 1x1 convolution to align the output dimensions with the input image's channel
    concat = Concatenate()([concat, Conv2D(64, (1, 1), activation='relu')])

    # Add a branch that merges the outputs of the main path and the branch through addition
    branch = Concatenate()([path1, path2])

    # Add two fully connected layers for classification
    x = Flatten()(concat)
    x = Dense(512, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model