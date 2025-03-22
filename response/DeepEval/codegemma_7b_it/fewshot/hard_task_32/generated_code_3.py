import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense, Concatenate

def block(input_tensor):
    """
    Defines a specialized block for the model.

    Args:
        input_tensor: The input tensor to the block.

    Returns:
        The output tensor of the block.
    """

    # Depthwise separable convolution
    conv_dw = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Dropout after depthwise convolution
    conv_dw = Dropout(0.2)(conv_dw)

    # 1x1 convolution to extract features
    conv_pw = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_dw)

    # Dropout after 1x1 convolution
    conv_pw = Dropout(0.2)(conv_pw)

    return conv_pw

def dl_model():
    """
    Constructs the deep learning model.

    Returns:
        The constructed Keras model.
    """

    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch_1 = block(input_tensor=input_layer)

    # Branch 2
    branch_2 = block(input_tensor=input_layer)

    # Branch 3
    branch_3 = block(input_tensor=input_layer)

    # Concatenate outputs from all branches
    concat = Concatenate()([branch_1, branch_2, branch_3])

    # Fully connected layer
    dense = Dense(units=128, activation='relu')(concat)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (assuming X_train and y_train are defined)
model.fit(X_train, y_train, epochs=10)

# Evaluate the model (assuming X_test and y_test are defined)
loss, accuracy = model.evaluate(X_test, y_test)

# Print the evaluation results
print('Loss:', loss)
print('Accuracy:', accuracy)