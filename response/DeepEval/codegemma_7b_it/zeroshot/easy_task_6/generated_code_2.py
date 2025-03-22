from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, add, Flatten, Dense, Dropout

def dl_model():

    # Input layer
    input_img = Input(shape=(28, 28, 1))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)

    # Branch path
    y = Conv2D(32, (3, 3), activation='relu')(input_img)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Combine paths
    combined = add([x, y])

    # Flatten and fully connected layer
    flattened = Flatten()(combined)
    output = Dense(10, activation='softmax')(flattened)

    # Model creation
    model = Model(inputs=input_img, outputs=output)

    return model