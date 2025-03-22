from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First path
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D((2, 2))(conv1)

    # Second path
    conv2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    maxpool2 = MaxPooling2D((2, 2))(conv2)

    # Combine paths
    combined_layer = Add()([maxpool1, maxpool2])

    # Flatten and classify
    flat = Flatten()(combined_layer)
    output = Dense(10, activation='softmax')(flat)

    model = Model(inputs=input_layer, outputs=output)
    return model