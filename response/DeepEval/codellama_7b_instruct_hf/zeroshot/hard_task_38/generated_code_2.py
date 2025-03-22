import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first processing pathway
    pathway1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    pathway1 = BatchNormalization()(pathway1)
    pathway1 = Conv2D(32, (3, 3), activation='relu')(pathway1)
    pathway1 = BatchNormalization()(pathway1)
    pathway1 = Conv2D(32, (3, 3), activation='relu')(pathway1)
    pathway1 = UpSampling2D()(pathway1)

    # Define the second processing pathway
    pathway2 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    pathway2 = BatchNormalization()(pathway2)
    pathway2 = Conv2D(32, (3, 3), activation='relu')(pathway2)
    pathway2 = BatchNormalization()(pathway2)
    pathway2 = Conv2D(32, (3, 3), activation='relu')(pathway2)
    pathway2 = UpSampling2D()(pathway2)

    # Define the merging layer
    merge_layer = Concatenate()([pathway1, pathway2])

    # Define the classifier
    classifier = Flatten()(merge_layer)
    classifier = Dense(128, activation='relu')(classifier)
    classifier = Dense(10, activation='softmax')(classifier)

    # Define the model
    model = Model(inputs=input_shape, outputs=classifier)

    return model