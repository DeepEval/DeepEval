from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the depthwise separable convolutional layer
    depthwise_separable_conv = DepthwiseSeparableConv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    )

    # Define the 1x1 convolutional layer
    conv = Conv2D(
        filters=32,
        kernel_size=1,
        padding="same",
        activation="relu"
    )

    # Define the dropout layer
    dropout = Dropout(rate=0.2)

    # Define the fully connected layer
    dense = Dense(units=10, activation="softmax")

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the model
    x = depthwise_separable_conv(input_layer)
    x = conv(x)
    x = dropout(x)
    x = Flatten()(x)
    x = dense(x)
    model = Model(inputs=input_layer, outputs=x)

    return model


# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))