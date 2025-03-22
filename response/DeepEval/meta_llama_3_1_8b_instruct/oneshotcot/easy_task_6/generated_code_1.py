import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Step 1: add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Step 3: branch path
    branch_path = input_layer

    # Step 4: combine two paths
    combined = Add()([conv2, branch_path])

    # Step 5: add flatten layer
    flatten_layer = Flatten()(combined)

    # Step 6: add dense layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Step 7: add output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Step 8: construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# call the function to get the model
model = dl_model()