import keras
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, Add

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10)(x)
    outputs = Add()([inputs, x])
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model