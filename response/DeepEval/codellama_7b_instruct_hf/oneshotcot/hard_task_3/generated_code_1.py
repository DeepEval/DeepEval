import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model
from keras.applications import VGG16

def dl_model():
    # Load pre-trained VGG16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    split_input = Lambda(lambda x: tf.split(x, 3, axis=3))(vgg.output)

    # Process each group through a sequence of 1x1 and 3x3 convolutions
    conv1 = Conv2D(64, (1, 1), activation='relu')(split_input[0])
    conv2 = Conv2D(64, (3, 3), activation='relu')(split_input[0])
    conv3 = Conv2D(64, (1, 1), activation='relu')(split_input[1])
    conv4 = Conv2D(64, (3, 3), activation='relu')(split_input[1])
    conv5 = Conv2D(64, (1, 1), activation='relu')(split_input[2])
    conv6 = Conv2D(64, (3, 3), activation='relu')(split_input[2])

    # Add dropout layer to mitigate overfitting
    dropout = Dropout(0.2)(Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6]))

    # Concatenate outputs from main and branch pathways
    concat = Concatenate()([dropout, conv1, conv2, conv3, conv4, conv5, conv6])

    # Pass concatenated output through a fully connected layer to complete the classification process
    flatten = Flatten()(concat)
    dense = Dense(10, activation='softmax')(flatten)

    # Create and return the model
    model = Model(inputs=vgg.input, outputs=dense)
    return model