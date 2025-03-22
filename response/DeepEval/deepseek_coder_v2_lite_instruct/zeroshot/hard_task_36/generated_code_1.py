import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Define the main pathway
    main_input = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_input)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # Define the branch pathway
    branch_input = Input(shape=(28, 28, 1))
    branch_output = Conv2D(32, kernel_size=(3, 3), activation='relu')(branch_input)

    # Fuse the pathways
    fused = Add()([x, branch_output])

    # Final layers
    x = GlobalAveragePooling2D()(fused)
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=output)

    return model