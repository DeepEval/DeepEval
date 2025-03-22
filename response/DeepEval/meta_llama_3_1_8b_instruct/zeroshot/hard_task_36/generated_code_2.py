# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the main pathway
    main_pathway = Input(shape=input_shape, name='main_pathway')
    
    # Convolutional layer to extract spatial features
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(main_pathway)
    
    # 1x1 convolutional layers to integrate inter-channel information
    x = Conv2D(32, kernel_size=1, activation='relu')(x)
    x = Conv2D(32, kernel_size=1, activation='relu')(x)
    
    # Max pooling to reduce the size of feature maps
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Dropout to mitigate overfitting
    x = Dropout(0.5)(x)

    # Define the branch pathway
    branch_pathway = Input(shape=input_shape, name='branch_pathway')
    x_branch = Conv2D(32, kernel_size=3, activation='relu')(branch_pathway)

    # Fusion of the two pathways
    x = Add()([x, x_branch])

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Flattening layer
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[main_pathway, branch_pathway], outputs=outputs)

    return model