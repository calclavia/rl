import tensorflow as tf


class ACModel:
    """
    Holds the AC model and Keras model that has been passed in.
    Compiles the policy and value loss functions.
    """

    def __init__(self, model_builder, beta):
        # Entropy weight
        self.beta = beta

        self.model = model_builder()
        # Output layers for policy and value estimations
        self.policies = self.model.outputs[:-1]
        self.value = self.model.outputs[-1]

    def compile(self, optimizer, grad_clip):
        # Only the worker network need ops for loss functions and gradient
        # updating.
        self.target_v = tf.placeholder(
            tf.float32, [None],  name='target_values')
        self.advantages = tf.placeholder(
            tf.float32, [None],  name='advantages')

        # Action chosen for every single policy output
        self.actions = []
        policy_losses = []
        entropies = []

        # Every policy output
        for policy in self.policies:
            num_actions = policy.get_shape()[1]
            action = tf.placeholder(tf.int32, [None])
            actions_hot = tf.one_hot(action, num_actions)
            self.actions.append(action)

            responsible_outputs = tf.reduce_sum(policy * actions_hot, [1])
            # Entropy regularization
            # TODO: Clipping should be configurable
            entropies.append(-tf.reduce_sum(policy *
                                            tf.log(tf.clip_by_value(policy, 1e-20, 1.0))))
            # Policy loss
            policy_losses.append(-tf.reduce_sum(tf.log(responsible_outputs)
                                                * self.advantages))

        # Compute average policy and entropy loss
        self.policy_loss = tf.reduce_sum(policy_losses)
        self.entropy = tf.reduce_sum(entropies)

        # Value loss (Mean squared error)
        self.value_loss = tf.reduce_mean(
            tf.square(self.target_v - tf.reshape(self.value, [-1])))
        # Learning rate for Critic is half of Actor's, so multiply by 0.5
        self.loss = 0.5 * self.value_loss + self.policy_loss - self.beta * self.entropy

        # Get gradients from local network using local losses
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)
        # Clip norm of gradients
        if grad_clip > 0:
            grads, self.grad_norms = tf.clip_by_global_norm(
            self.gradients, grad_clip)
        else:
            grads = self.gradients
            
        # Apply local gradients to global network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.train = optimizer.apply_gradients(zip(grads, global_vars))
