from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the main pathway
    main_pathway = Conv2D(32, (3, 3), activation='relu')(input_shape)
    main_pathway = Conv2D(32, (3, 3), activation='relu')(main_pathway)
    main_pathway = MaxPooling2D((2, 2))(main_pathway)
    main_pathway = Dropout(0.5)(main_pathway)

    # Define the branch pathway
    branch_pathway = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_pathway = Conv2D(32, (3, 3), activation='relu')(branch_pathway)
    branch_pathway = MaxPooling2D((2, 2))(branch_pathway)

    # Define the fusion layer
    fusion = Add()([main_pathway, branch_pathway])

    # Define the global average pooling layer
    global_average_pooling = GlobalAveragePooling2D()(fusion)

    # Define the flatten layer
    flatten = Flatten()(global_average_pooling)

    # Define the fully connected layer
    fully_connected = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_shape, outputs=fully_connected)

    return model