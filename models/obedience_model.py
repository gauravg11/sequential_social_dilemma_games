import numpy as np
import tensorflow as tf

from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override


class ObedienceLSTM(RecurrentTFModelV2):
    """An LSTM with two heads, one for taking actions and one for producing symbols."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        super(ObedienceLSTM, self).__init__(obs_space, action_space, num_outputs,
                                            model_config, name)

        self._value_out = -1
        self.obs_space = obs_space
        vision_space = self.obs_space.original_space.spaces[0]
        message_space = self.obs_space.original_space.spaces[1]

        # The inputs of the shared trunk. We will concatenate the observation space with shared info about the
        # visibility of agents. Currently we assume all the agents have equally sized action spaces.
        self.num_outputs = num_outputs
        self.num_agents = model_config["custom_options"]["num_agents"]
        self.num_symbols = model_config["custom_options"]["num_symbols"]
        self.cell_size = model_config["custom_options"].get("cell_size")

        # an extra none for the time dimension
        inputs = tf.keras.layers.Input(
            shape=(None,) + vision_space.shape, name="observations")

        # Build the CNN layer
        last_layer = inputs
        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                channels_last=True,
                name="conv{}".format(i)))(last_layer)
        out_size, kernel, stride = filters[-1]
        if len(filters) == 1:
            i = -1

        # should be batch x time x height x width x channel
        conv_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="valid",
            name="conv{}".format(i + 1)))(last_layer)

        flat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(conv_out)

        # Add the fully connected layers
        hiddens = model_config["custom_options"].get("fcnet_hiddens")
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        last_layer = flat_layer
        i = 1
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}_{}".format(i, name),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)  # ME: string custom initializer
            i += 1

        messages_layer = tf.keras.layers.Input(shape=((None,) + message_space.shape), name="messages")
        last_layer = tf.keras.layers.concatenate([last_layer, messages_layer])

        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm")(
            inputs=last_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])

        num_actions_out = action_space.nvec[0]
        num_messages_out = self.num_outputs - num_actions_out

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            num_actions_out,
            activation=tf.keras.activations.linear,
            name=name)(lstm_out)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(lstm_out)

        message_logits = tf.keras.layers.Dense(
            num_messages_out,
            activation=tf.keras.activations.linear,
            name=f'message_{name}')(lstm_out)

        inputs = [inputs, messages_layer, seq_in, state_in_h, state_in_c]
        self.rnn_model = tf.keras.Model(
            inputs=inputs,
            outputs=[logits, value_out, message_logits, state_h, state_c])

        self.register_variables(self.rnn_model.variables)
        # self.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()"""

        # first we add the time dimension for each object
        input_dict["obs_vision"] = add_time_dimension(input_dict["obs"][0], seq_lens)
        input_dict["obs_messages"] = add_time_dimension(input_dict["obs"][1], seq_lens)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)

        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        # we operate on our obs, others previous actions, our previous actions, our previous rewards
        # TODO(@evinitsky) are we passing seq_lens correctly? should we pass prev_actions, prev_rewards etc?

        model_out, self._value_out, model_out_messages, h, c = self.rnn_model([
                                                                          input_dict['obs_vision'],
                                                                          input_dict['obs_messages'],
                                                                          seq_lens]
                                                                      + state)

        return tf.concat([model_out, model_out_messages], axis=-1), [h, c]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]