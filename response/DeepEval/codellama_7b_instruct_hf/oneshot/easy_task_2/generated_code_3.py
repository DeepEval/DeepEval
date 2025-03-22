from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the first sequential feature extraction layer
    conv1 = Conv2D(32, (3, 3), activation='relu')
    maxpool1 = MaxPooling2D((2, 2))
    sequential_feature_extraction_layer1 = layers.concatenate([conv1, maxpool1], axis=1)

    # Define the second sequential feature extraction layer
    conv2 = Conv2D(64, (3, 3), activation='relu')
    maxpool2 = MaxPooling2D((2, 2))
    sequential_feature_extraction_layer2 = layers.concatenate([conv2, maxpool2], axis=1)

    # Flatten the feature maps
    flatten_layer = Flatten()

    # Define the first fully connected layer
    dense1 = Dense(128, activation='relu')

    # Define the second fully connected layer
    dense2 = Dense(64, activation='relu')

    # Define the output layer
    output_layer = Dense(10, activation='softmax')

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)
    model.add(sequential_feature_extraction_layer1)
    model.add(sequential_feature_extraction_layer2)
    model.add(flatten_layer)
    model.add(dense1)
    model.add(dense2)
    model.add(output_layer)

    return model