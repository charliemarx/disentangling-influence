"""
Generates a disentangled representation.
Author: -----
"""

import keras.backend as K
from keras.layers import Dense, Concatenate, Lambda, Input
from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.optimizers import Adam, SGD

import numpy as np
import pandas as pd

from DisentanglingInfluence.utils import *
from DisentanglingInfluence.Disentangling.model_factories import model_factory

def disentangle(data_generator, n_feats, n_protected,
            batch_size=16, train_steps=8000,
            latent_dim=5, stoch_dim=0,
            dec_weight=1, disc_weight=0.05,
            output_dir=".",
            encoder_arch=model_factory.Encoder,
            decoder_arch=model_factory.Decoder,
            disc_arch=model_factory.Discriminator,
            enc_layer_sizes=[30,20,10],
            dec_layer_sizes=[30,20,10],
            disc_layer_sizes=[30,20,10],
            dec_final_activ=None,
            disc_final_activ=None,
            disc_metrics=["acc", binary_crossentropy],
            verbose=True,
            show_train_history=False,
            save_models=False,
            return_loss=False,
            learning_rate=0.01, categorical_protected_feature=False,
            learning_rate_encoder=None):


    ####### miscellaneous model utilities ############
    # Toggle the trainability of a model
    def toggle_trainable(model, trainable):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    # calculates binary crossentropy between y_pred[i] and 0 where i is the index of the true class
    def strong_competitive_CE(y_true, y_pred):
        target_prob = K.batch_dot(y_true, y_pred, axes=(1,1))
        return(binary_crossentropy(target_prob, K.zeros(shape=K.shape(target_prob))))

    # def negative_MAE(y_true, y_pred):
    # 	return(-mean_absolute_error(y_true, y_pred))
    def negative_MSE(y_true, y_pred):
        return(-mean_squared_error(y_true, y_pred))

    def asymptotic_negative_MAE(y_true, y_pred):
        x_abs = mean_absolute_error(y_true, y_pred)
        loss = (x_abs + 1)**(-3)
        return(loss)

    ########## design the architecture ##################
    # symbolic tensors for data inputs
    feat_input = Input(shape=(n_feats,), name="feat_input")
    latent_input = Input(shape=(latent_dim,), name="latent_input")
    protected_input = Input(shape=(n_protected,), name="protected_input")

    # instantiate model builders
    EncoderBuilder = encoder_arch(feat_input, latent_dim)
    DecoderBuilder = decoder_arch(latent_input, protected_input, n_feats)
    DiscriminatorBuilder = disc_arch(latent_input, n_protected)

    # individual model graphs
    latent = EncoderBuilder.build(layer_sizes=enc_layer_sizes)
    dec_output = DecoderBuilder.build(layer_sizes=dec_layer_sizes, final_activation=dec_final_activ)
    disc_output = DiscriminatorBuilder.build(layer_sizes=disc_layer_sizes, final_activation=disc_final_activ)

    # individual models
    Enc = Model(feat_input, latent, name="Enc")
    Dec = Model([latent_input, protected_input], dec_output, name="Dec")
    Disc = Model(latent_input, disc_output, name="Disc")

    # symbolic tensors for output of models with intermediate inputs
    x_hat = Dec([latent, protected_input])
    Disc_pred = Disc([latent])

    # combined models
    AutoEncoder = Model(inputs=[feat_input, protected_input], outputs=x_hat, name="AE")
    FullModel = Model(inputs=[feat_input, protected_input],
                outputs=[x_hat, Disc_pred],
                name="FullModel")

    # compile models we may want to train, taking care to consider which weights are trainable in each model.
    toggle_trainable(Enc, trainable=False)
    toggle_trainable(Dec, trainable=False)


    sgd = SGD(lr=learning_rate, decay=1e-5, momentum=0.9, nesterov=True)

    if categorical_protected_feature:
        Disc.compile(optimizer=sgd, loss="binary_crossentropy", metrics=disc_metrics)
    else:
        Disc.compile(optimizer=sgd, loss="mean_squared_error", metrics=disc_metrics)

    if learning_rate_encoder != None:
        sgd = SGD(lr=learning_rate_encoder, decay=0, momentum=0.9, nesterov=True)

    toggle_trainable(Enc, trainable=True)
    toggle_trainable(Dec, trainable=True)
    toggle_trainable(Disc, trainable=False)

    AutoEncoder.compile(optimizer=sgd, loss="mse")
    FullModel.compile(optimizer=sgd,
            loss={"Dec":"mse", "Disc":negative_MSE},
            loss_weights={"Dec":dec_weight, "Disc":disc_weight})

    # print summary of architecture
    # FullModel.summary()

    # logs for training
    G_losses = []
    recon_losses = []
    disc_losses = []
    disc_accs = []
    disc_BCRs = []

    # training loop

    for i in range(train_steps):
        # generate inputs
        x_batch, p_batch = next(data_generator)
        latent = Enc.predict(x_batch)

        # train AE
        G_loss, recon_loss, _ = FullModel.train_on_batch(x=[x_batch, p_batch], y=[x_batch, p_batch])

        # train Discriminator
        disc_loss, disc_acc, disc_BCR = Disc.train_on_batch(x=[latent], y=[p_batch])

        # record losses
        G_losses.append(G_loss)
        recon_losses.append(recon_loss)
        disc_losses.append(disc_loss)
        disc_accs.append(disc_acc)
        disc_BCRs.append(disc_BCR)


        if verbose and i % 100 == 0:
            print("Step: {:>5} -- G_loss: {:.3f} -- recon_loss: {:.3f} -- D_loss: {:.3f} -- D_acc: {:.3f} -- D_BCR: {:.3f}"
                    .format(i, np.mean(G_losses[-100:]), np.mean(recon_losses[-100:]),
                    np.mean(disc_losses[-100:]), np.mean(disc_accs[-100:]),
                    np.mean(disc_BCRs[-100:])))

    # data_generator.set_mode(tuple)
    # print(next(data_generator))
    # print(FullModel.evaluate_generator(data_generator, steps=200))
    if show_train_history:
        index_plot(G_losses, filename=output_dir+"AE_loss.png", x_lab="train step", y_lab="loss", title="AE loss")
        index_plot(recon_losses, filename=output_dir+"recon_loss.png", x_lab="train step", y_lab="loss", title="Reconstruction loss")
        index_plot(disc_losses, filename=output_dir+"disc_loss.png", x_lab="train step", y_lab="loss", title="Discriminator loss")
        index_plot(disc_BCRs, filename=output_dir+"disc_BCR_loss.png", x_lab="train step", y_lab="BCR", title="Discriminator loss")


    if save_models:
        FullModel.save(output_dir+"FullModel.h5")
        Enc.save(output_dir+"Encoder.h5")
        Dec.save(output_dir+"Decoder.h5")
        AutoEncoder.save(output_dir+"AutoEncoder.h5")

    if return_loss:
        return (G_losses[-1], recon_losses[-1], disc_losses[-1], disc_BCRs)
    else:
        return(FullModel, Enc, Dec, Disc, AutoEncoder)



