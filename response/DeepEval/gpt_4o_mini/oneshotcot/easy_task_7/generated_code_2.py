import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Step 1: Input Layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Main Path - First Convolution + Dropout
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)

    # Step 3: Main Path - Second Convolution + Dropout
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)

    # Step 4: Main Path - Restore Channel Count
    conv3 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)

    # Step 5: Branch Path (Direct connection to input)
    branch_path = input_layer

    # Step 6: Combine Paths using Addition
    combined = Add()([conv3, branch_path])

    # Step 7: Flattening Layer
    flatten_layer = Flatten()(combined)

    # Step 8: Fully Connected Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model