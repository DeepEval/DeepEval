import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_pool = MaxPooling2D(pool_size=(2, 2))(main_conv)
    main_avg_pool = GlobalAveragePooling2D()(main_pool)
    main_fc = Dense(units=128, activation='relu')(main_avg_pool)

    # Branch path
    branch_input = input_layer
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_input)

    # Combine main and branch paths
    combined = Add()([main_avg_pool, branch_conv])
    combined = Dense(units=128, activation='relu')(combined)
    output = Dense(units=10, activation='softmax')(combined)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])