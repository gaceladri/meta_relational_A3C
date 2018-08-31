import tensorflow as tf
import numpy as np

import scipy.signal
import json
import os

import threading
import multiprocessing

from memory import RelationalMemory
from PIL import Image, ImageDraw, ImageFont


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class contextual_bandit():
    def __init__(self):
        self.num_actions = 2
        self.reset()

    def get_state(self):
        self.internal_state = np.random.permutation(self.choices)
        self.state = np.concatenate(np.reshape(
            self.internal_state, [2, 1, 1, 3]), axis=1)
        return self.state

    def reset(self):
        self.timestep = 0
        color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]
        a = [np.reshape(np.array(color), [1, 1, 3]),
             np.reshape(1-np.array(color), [1, 1, 3])]
        self.true = a[0]
        self.choices = a
        return self.get_state()

    def pullArm(self, action):
        self.timestep += 1
        if (self.internal_state[action] == self.true).all() == True:
            reward = 1.0
        else:
            reward = 0.0
        new_state = self.get_state()
        if self.timestep > 99:
            done = True
        else:
            done = False
        return new_state, reward, done, self.timestep


class Agent():
    def __init__(self, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input placeholders
            self.state = tf.placeholder(
                shape=[None, 1, 2, 3], dtype=tf.float32)
            self.prev_rewards = tf.placeholder(
                shape=[None, 1], dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(
                self.prev_actions, a_size, dtype=tf.float32)

            # Recurrent network for temporal dependencies
            hidden = tf.concat([tf.layers.flatten(self.state),
                                self.prev_rewards, self.prev_actions_onehot, self.timestep], 1)
            relational_cell = RelationalMemory(mem_slots=2048, head_size=12, num_heads=1, num_blocks=1,
                                               forget_bias=1.0, input_bias=0.0, gate_style='unit', attention_mlp_layers=2, key_size=None, name='relational_memory')
            # state_init = relational_cell.initial_state(batch_size=1, trainable=False)
            state_init = np.eye(relational_cell._mem_slots, dtype=np.float32)
            state_init = state_init[np.newaxis, ...]
            state_init = state_init[:, :, :relational_cell._mem_size]
            self.state_init = state_init
            self.state_in = tf.placeholder(
                tf.float32, shape=[1, relational_cell._mem_slots, relational_cell._mem_size])
            step_size = tf.shape(self.prev_rewards)[:1]
            rnn_in = tf.expand_dims(hidden, [0])

            output_sequence, cell_state = tf.nn.dynamic_rnn(
                relational_cell, rnn_in, sequence_length=step_size, initial_state=self.state_init,
                time_major=False)
            self.state_out = cell_state
            print('output_sequence.shape')
            print(output_sequence.shape)
            print('output_sequence.shape')
            rnn_out = tf.reshape(output_sequence, [-1, 12])

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, a_size, dtype=tf.float32)

            # Output layer for policy and value estimations
            self.policy = tf.contrib.layers.fully_connected(rnn_out, a_size,
                                                            activation_fn=tf.nn.softmax,
                                                            weights_initializer=normalized_columns_initializer(
                                                                0.01),
                                                            biases_initializer=None)
            self.value = tf.contrib.layers.fully_connected(rnn_out, 1,
                                                           activation_fn=None,
                                                           weights_initializer=normalized_columns_initializer(
                                                               1.0),
                                                           biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(
                    shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(
                    self.policy * self.actions_onehot, [1])

                # Loss functions
                print(tf.shape(self.value))
                print('tf.shape(self.value)')
                self.value_loss = 0.5 * \
                    tf.reduce_sum(tf.square(self.target_v -
                                            tf.reshape(self.value, [-1])))

                self.entropy = - \
                    tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = - \
                    tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7))

                # (learning rate for Critic is half of Actor's, so multiply by 0.5)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(
                    self.gradients, 999.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))


class Worker():
    def __init__(self, game, name, a_size, trainer, model_path, global_episodes):
        self.name = "worker" + str(name)
        self.number = name
        self.model_path = model_path
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = Agent(a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 5]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.state: np.stack(states, axis=0),
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.timestep: np.vstack(timesteps),
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in: rnn_state}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver, train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker" + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0
                s = self.env.reset()
                rnn_state = self.local_AC.state_init

                while d == False:
                    # Take an action using probabilities from policy networks output.
                    a_dist, v, rnn_state_new = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                                        feed_dict={
                        self.local_AC.state: [s],
                        self.local_AC.prev_rewards: [[r]],
                        self.local_AC.timestep: [[t]],
                        self.local_AC.prev_actions: [a],
                        self.local_AC.state_in: rnn_state})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new
                    s1, r, d, t = self.env.pullArm(a)
                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append([v[0, 0]])
                    episode_reward += r
                    total_steps += 1
                    episode_step_count += 1
                    s = s1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l, p_l, e_l, g_n, v_n = self.train(
                        episode_buffer, sess, gamma, 0.0)

                # Periodically save gifts of episodes, model parameters, and summary statistics.
                if episode_count % 20 == 0 and episode_count != 0:
                    if episode_count % 500 == 0 and self.name == 'worker_0' and train == True and len(self.episode_rewards) != 0:
                        saver.save(sess, self.model_path+'/model' +
                                   str(episode_count) + '.cptk')
                        print("Saved model")

                    mean_reward = np.mean(self.episode_rewards[-10:])
                    mean_length = np.mean(self.episode_lengths[-10:])
                    mean_value = np.mean(self.episode_mean_values[-10:])
                    summary = tf.summary
                    summary.scalar('Perf/Reward', mean_reward)
                    summary.scalar('Perf/Length', mean_length)
                    summary.scalar('Perf/Value', mean_value)
                    if train == True:
                        summary.scalar('Losses/Value Loss', v_l)
                        summary.scalar('Losses/Policy Loss', p_l)
                        summary.scalar('Losses/Entropy', e_l)
                        summary.scalar('Losses/Grad Norm', g_n)
                        summary.scalar('Losses/Var Norm', v_n)
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


def main(args):
    tf.reset_default_graph()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        f_log = open(os.path.join(args.save_dir, 'print.log'), 'w')

    def lprint(*a, **kw):
        print(*a, **kw)
        print(*a, **kw, file=f_log)

    lprint('input args:\n', json.dumps(vars(args), indent=4,
                                       separators=(',', ':')))

    gamma = 0.8
    a_size = 2
    train = args.train
    load_model = args.load_params

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(
            0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
        master_network = Agent(a_size, 'global', None)
        num_workers = multiprocessing.cpu_count()
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(contextual_bandit(), i, a_size,
                                  trainer, args.save_dir, global_episodes))
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print("Loading Model...")
            ckpt = tf.train.get_checkpoint_state(args.save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        all_params = tf.trainable_variables()
        lprint('# of Parameters', sum(np.prod(p.get_shape().as_list())
                                      for p in all_params))

        worker_threads = []
        for worker in workers:
            def worker_work(): return worker.work(gamma, sess, coord, saver, train)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)


if __name__ == '__main__':
    import argparse
    import datetime
    import dateutil.tz
    import functools
    import os.path as osp

    parser = argparse.ArgumentParser()

    # data I/O
    parser.add_argument('-o', '--save_dir', type=str, default='./save',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-t', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=False,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('-g', '--train', type=bool, default=True,
                        help='Train or test mode. If True it will train the model')

    # optimization
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=1e-3, help='Base learning rate')

    FLAGS = parser.parse_args()

    timestamp = datetime.datetime.now(
        dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    logdir = 'Meta_learner_%s' % (timestamp)

    FLAGS.save_dir = osp.join(FLAGS.save_dir, logdir)
    main(FLAGS)
