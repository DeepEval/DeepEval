import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)
    
    conv3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(dropout2)

    # Branch path
    branch_path = input_layer

    # Combine both paths
    combined = Add()([conv3, branch_path])

    # Final layers
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model