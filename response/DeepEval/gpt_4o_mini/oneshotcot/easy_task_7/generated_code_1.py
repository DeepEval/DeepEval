import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, Add

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Main path - first convolution and dropout block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.25)(conv1)

    # Step 3: Main path - second convolution and dropout block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv2)

    # Step 4: Main path - convolution to restore number of channels
    conv3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(dropout2)

    # Step 5: Branch path - connects directly to the input
    branch_path = input_layer

    # Step 6: Combine outputs from both paths using addition
    combined_output = Add()([conv3, branch_path])

    # Step 7: Add flatten layer
    flatten_layer = Flatten()(combined_output)

    # Step 8: Add fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 9: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model