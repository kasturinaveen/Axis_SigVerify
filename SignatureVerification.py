from keras.layers import Input, Lambda
from keras.models import load_model, Model
from sigverification import create_base_network_signet, euclidean_distance, eucl_dist_output_shape, contrastive_loss
from keras.optimizers import SGD, RMSprop, Adadelta


def build_model():
    input_shape = (155,220,1)
    base_network = create_base_network_signet(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)
    rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
    # adadelta = Adadelta()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics= ['accuracy'])
    model.load_weights("model_prep_acc.h5")
    return model
