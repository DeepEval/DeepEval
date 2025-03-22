import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Add, Reshape, Flatten
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First Block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Main path combines input and output of first block via addition
    skip_connection = Add()([input_layer, avg_pool1])

    # Second Block
    global_avg_pool = GlobalAveragePooling2D()(skip_connection)
    channel_weights = Dense(units=32, activation='relu')(global_avg_pool)
    channel_weights = Dense(units=32, activation='sigmoid')(channel_weights)

    # Reshaping to multiply with input
    reshaped_weights = Reshape((1, 1, 32))(channel_weights)
    multiplied = Multiply()([input_layer, reshaped_weights])  # Multiply input with channel weights

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(multiplied)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# To create the model, you can call:
model = dl_model()
model.summary()  # Optional: Print the model summary to see the architecture