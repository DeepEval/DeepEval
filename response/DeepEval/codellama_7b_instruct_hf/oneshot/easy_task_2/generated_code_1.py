import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.regularizers import l2

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(224, 224, 3))

    # Define the first sequential feature extraction layer
    conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Define the second sequential feature extraction layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Define the third sequential feature extraction layer
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Flatten the feature maps
    flatten = Flatten()(pool3)

    # Add dropout layers to mitigate overfitting
    drop1 = Dropout(0.2)(flatten)
    drop2 = Dropout(0.3)(drop1)

    # Define the fully connected layers
    dense1 = Dense(512, activation='relu')(drop2)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(1000, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model