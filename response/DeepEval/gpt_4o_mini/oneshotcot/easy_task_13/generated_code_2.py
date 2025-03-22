import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Two 1x1 convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)

    # Step 3: 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)
    dropout3 = Dropout(0.25)(conv3)

    # Step 4: 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout3)
    dropout4 = Dropout(0.25)(conv4)

    # Step 5: 1x1 convolutional layer to restore channel count
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout4)

    # Step 6: Combine processed features with original input via addition
    added_output = Add()([input_layer, conv5])

    # Step 7: Add flattening layer
    flatten_layer = Flatten()(added_output)
    
    # Step 8: Add fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model