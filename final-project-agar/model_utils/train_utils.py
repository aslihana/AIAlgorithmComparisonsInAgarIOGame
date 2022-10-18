from operator import add, and_
import numpy as np
from functools import reduce
import math
import matplotlib.pyplot as plt
import torch

from gamestate import GameState
import utils
import config as conf
import model_utils.fs_utils as fs
import xlsxwriter

from typing import List, TypeVar

T = TypeVar("T")
data = []




# below are helper functions for plotting

def get_means_over_window(vals, window_size):
    means = []
    for i in range(len(vals)):
        if i < window_size - 1:
            means.append(np.mean(vals[0:(i + 1)]))
        else:
            means.append(np.mean(vals[(i - window_size + 1):(i + 1)]))
    return means



# helper function to calculate epsilon decay factor according to starting & finishing epsilon, and how many
#   episode to decay over


def get_epsilon_decay_factor(e_max, e_min, e_decay_window):
    return math.exp(math.log(e_min / e_max) / e_decay_window)


# helper function used to iterate through all the models and select an action


def select_model_actions(models, state):
    model_actions = []
    for model in models:
        model_actions.append(model.get_action(state))
    return model_actions


# helper function used to iterate through all the models and train them


def optimize_models(models):
    for model in models:
        model.optimize()


# helper function used to iterate through all the models and update their replay buffer


def update_models_memory(models, state, actions, next_state, rewards, dones):
    for (model, action, reward, done) in zip(models, actions, rewards, dones):
        model.remember(state, action, next_state, reward, done)


def train_models(
        env,
        model,
        enemies,
        episodes=10,
        steps=2500,
        print_every=200,
        model_name="train_drl",
        mean_window=10,
        target_update=10,
        num_checkpoints=1):
    print("TRAIN MODE")

    training_losses = []
    training_rewards = []
    models = [model] + enemies
    maxes = 0
    workbook = xlsxwriter.Workbook('dqn_trainingScores.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    row2 = 1
    col = 0
    worksheet.write(row, col, 'Episode No')
    worksheet.write(row, col + 1, 'Step Score')
    worksheet.write(row, col + 4, 'Episode No')
    worksheet.write(row, col + 5, 'Max Epd. Score')
    worksheet.write(row, col + 6, 'Mean Epsd.Score')
    worksheet.write(row, col + 7, 'Cumulative Mean Score')
    worksheet.write(row, col + 8, 'Mean Epsd.Reward')
    worksheet.write(row, col + 9, 'Mean Epsd.Loss')
    worksheet.write(row, col + 10, 'Epsilon')
    row += 1

    for episode in range(episodes):
        epsilon = 1.0  # Hardcode to 1 until we start decaying when replay buffer is full enough
        average=0
        # reset the environment to fresh starting state with game agents initialized for models
        episode_rewards = [0 for _ in models]
        episode_loss = []
        score_lst = []

        for m in models:
            m.done = False
            m.eval = False

        env.reset(models)
        state = env.get_state()  # get the first state
        total_episode_scores=0

        for step in range(steps):  # cap the num of game ticks
            actions = select_model_actions(models, state)
            (agents, foods, _viruses, _masses, _time) = state
            agent = agents[model.id]
            score = agent.max_mass - agent.starting_mass
            print(score)
            worksheet.write(row, col, episode)
            worksheet.write(row, col+1, score)
            row += 1
            score_lst.append(score)
            total_episode_scores = total_episode_scores + score
            # environment determines new state, reward, whether terminal, based on actions taken by all models
            rewards, dones = env.update_game_state(models, actions)
            next_state = env.get_state()

            episode_rewards = list(
                map(add, episode_rewards, rewards))  # update rewards

            # update replay memory
            update_models_memory(
                models, state, actions, next_state, rewards, dones)

            # optimize models
            loss = model.optimize()
            episode_loss.append(loss)

            # check for termination of our player
            if dones[0]:
                break

            # terminate if all other players are dead
            if (len(dones) > 1):
                if reduce(and_, dones[1:]):
                    break

            state = next_state  # update the state

            if step % print_every == 0 and step != 0:
                print("----STEP %s----" % step)
                for idx, model in enumerate(models):
                    print("Model %s | Reward: %s\tLoss: %s" %
                          (model.id, episode_rewards[idx], episode_loss[idx]))
        average=sum(score_lst) / len(score_lst)
        max_score= max(score_lst)
        maxes = maxes + max_score
        worksheet.write(row2, col + 4, episode)
        worksheet.write(row2, col + 5, max_score)
        worksheet.write(row2, col + 6, average)
        worksheet.write(row2, col + 7, maxes/(episode+1))

        # decay epsilon
        if model.learning_start:
            epsilon = model.decay_epsilon()

        # sync target net with policy
        if episode % target_update == 0:
            model.sync_target_net()


        episode_loss = [loss for loss in episode_loss if loss is not None]
        training_loss = np.mean(episode_loss) if len(episode_loss) else -1
        if training_loss >= 0:
            training_losses.append(training_loss)
        training_rewards.append(episode_rewards[0])
        worksheet.write(row2, col + 8, np.mean(training_rewards[-mean_window:]))
        worksheet.write(row2, col + 9, training_loss)
        worksheet.write(row2, col + 10, epsilon)
        row2 += 1
        print('{}\tMean Episode Loss: {:.4f}\tEpisode Reward: {:.4f}\tMean Reward: {:.4f}\tEpsilon: {:.4f}'.format(
            episode,
            training_loss,
            episode_rewards[0],
            np.mean(training_rewards[-mean_window:]),
            epsilon))
    workbook.close()



def test_models(env, model, enemies, steps=2500, print_every=200):
    print("TEST MODE")
    models = [model] + enemies
    episode_rewards = [0 for _ in models]
    for m in models:
        m.done = False
        m.eval = True

    env.reset(models)
    state = env.get_state()  # get the first state

    for step in range(steps):  # cap the num of game ticks
        actions = select_model_actions(models, state)

        # environment determines new state, reward, whether terminal, based on actions taken by all models
        rewards, dones = env.update_game_state(models, actions)

        next_state = env.get_state()

        episode_rewards = list(
            map(add, episode_rewards, rewards))  # update rewards

        # check for termination of our player
        if dones[0]:
            break

        state = next_state  # update the state

        if step % print_every == 0:
            print("----STEP %s rewards----" % step)
            for idx, model in enumerate(models):
                print("Model %s: %s" % (model.id, episode_rewards[idx]))
    print("------TEST rewards------")
    for idx, model in enumerate(models):
        print("Model %s: %s" % (model.id, episode_rewards[idx]))


def train_deepcnn_model(cnn_model, model_name, adversary_models, frame_skip=4,
                        update_freq=4, target_net_sync_freq=500, max_eps=200,
                        max_steps_per_ep=500, mean_window=10, prefill_buffer=False):
    env = GameState()
    workbook = xlsxwriter.Workbook('cnn_trainingScores.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    row2 = 1
    col = 0
    worksheet.write(row, col, 'Episode No')
    worksheet.write(row, col + 1, 'Step Score')
    worksheet.write(row, col + 2, 'Step Reward')
    worksheet.write(row, col + 4, 'Episode No')
    worksheet.write(row, col + 5, 'Max Epd. Score')
    worksheet.write(row, col + 6, 'Mean Epsd.Score')
    worksheet.write(row, col + 7, 'Cumulative Mean Score')
    worksheet.write(row, col + 8, 'Mean Epsd.Reward')
    worksheet.write(row, col + 9, 'Mean Epsd.Loss')
    row += 1
    # ensure the CNN model is centered in window
    if not cnn_model.camera_follow:
        raise ValueError('Camera needs to be following CNN')

    models = [cnn_model] + adversary_models
    training_losses = []
    training_rewards = []
    training_scores = []
    training_steps_survived = []

    start_time = utils.current_milli_time()

    # if specified, burn in the replay buffer to fill it with some examples before starting to train
    if prefill_buffer:
        print('Filling replay buffer to ' + str(cnn_model.replay_buffer.prefill_amt *
                                                100 / cnn_model.replay_buffer.capacity) + '% capacity...')
        env.reset(models)
        pixels = env.get_pixels()
        while cnn_model.replay_buffer.prefill_capacity() < 1.0:
            cnn_model.state_buffer.append(cnn_model.preprocess_state(pixels))

            actions = [utils.get_random_action() for m in models]
            rewards, dones = env.update_game_state(models, actions)

            next_pixels = env.get_pixels()
            cnn_model.next_state_buffer.append(
                cnn_model.preprocess_state(pixels))
            cnn_model.remember(
                pixels, actions[0], next_pixels, rewards[0], dones[0])

            if dones[0]:
                env.reset(models)
                pixels = env.get_pixels()
            else:
                pixels = next_pixels

        print('Replay buffer filled with ' +
              str(len(cnn_model.replay_buffer)) + ' samples!')
    else:
        print('Replay buffer prefill disabled.')

    print('Beginning training...')
    for ep in range(max_eps):
        env.reset(models)
        state = env.get_state()
        pixels = env.get_pixels()

        # used for frame skipping, to simply repeat the last action chosen
        action = None
        total_episode_scores=0
        update_losses = []
        ep_reward = 0
        cnn_agent = env.get_agent_of_model(cnn_model)
        print('=== Starting Episode %s ===' % ep)
        for step in range(max_steps_per_ep):
            score = cnn_agent.max_mass - cnn_agent.starting_mass
            print(score)
            worksheet.write(row, col, ep)
            worksheet.write(row, col + 1, score)
            total_episode_scores =total_episode_scores + score
            if cnn_model.step_count % 250 == 0:
                print('Step %s' % cnn_model.step_count)
            # append current pixel state to buffer
            cnn_model.state_buffer.append(cnn_model.preprocess_state(pixels))

            # only update the current action every FRAME_SKIP frames of the game
            if step % frame_skip == 0:
                with torch.no_grad():
                    cnn_model.net.eval()
                    # stack last tau frames to get CNN action based on them
                    s_0 = np.stack([cnn_model.state_buffer])
                    action = cnn_model.get_stacked_action(s_0)

            cnn_model.step_count += 1

            # get actions for other models
            actions = select_model_actions(adversary_models, state)

            # determine rewards and dones by letting all models take actions
            rewards, dones = env.update_game_state(models, [action] + actions)
            ep_reward += rewards[0]
            worksheet.write(row, col + 2, rewards[0])
            row += 1
            # get the next pixel state and append to next state buffer
            next_state = env.get_state()
            next_pixels = env.get_pixels()
            cnn_model.next_state_buffer.append(
                cnn_model.preprocess_state(next_pixels))

            # push to replay buffer
            cnn_model.remember(pixels, action, next_pixels,
                               rewards[0], dones[0])

            # optimize model
            if step % update_freq == 0:
                update_loss = cnn_model.optimize()
                cnn_model.net_update_count += 1
                if update_loss is not None:
                    update_losses.append(update_loss)

            # re-sync the target network to updated network every on function of number
            # of net parameter updates
            if cnn_model.net_update_count % target_net_sync_freq == 0:
                cnn_model.sync_target_net()

            if dones[0]:
                break

            # move state and pixels one step forward
            state = next_state
            pixels = next_pixels

        # at end of episode, decay epsilon
        if cnn_model.epsilon > cnn_model.end_epsilon:
            cnn_model.epsilon *= cnn_model.epsilon_decay_fac
        else:
            cnn_model.epsilon = cnn_model.end_epsilon

        # also keep track of episode mean update loss, episode score,
        # episode reward, and mean score/reward over last {mean_window} episodes
        training_losses.append(np.mean(update_losses))
        training_rewards.append(ep_reward)
        # subtract starting mass to get accurate count of maximum growth even with random init mass
        training_scores.append(cnn_agent.max_mass - cnn_agent.starting_mass)
        training_steps_survived.append(cnn_agent.steps_taken)
        worksheet.write(row2, col + 4, ep)
        worksheet.write(row2, col + 5, cnn_agent.max_mass - cnn_agent.starting_mass)
        worksheet.write(row2, col + 6, total_episode_scores/max_steps_per_ep)
        worksheet.write(row2, col + 7, np.mean(training_scores[-mean_window:]))
        worksheet.write(row2, col + 8, np.mean(training_rewards[-mean_window:]))
        worksheet.write(row2, col + 9, np.mean(update_losses))
        row2 += 1
        total_episode_scores=0

        print('Ep Score: {:.4f} | Mean Score: {:.4f} | Steps Survived: {:d} | Mean Steps Survived: {:.2f}'.format(
            cnn_agent.max_mass -
            cnn_agent.starting_mass, np.mean(training_scores[-mean_window:]),
            cnn_agent.steps_taken, np.mean(training_steps_survived[-mean_window:])))
        print('Mean Ep Loss: {:.4f} | Ep Reward: {:.4f} | Mean Reward: {:.4f}'.format(
            np.mean(update_losses), ep_reward, np.mean(training_scores[-mean_window:])))
        print('Model has been training for {:.4f} minutes.'.format(
            (utils.current_milli_time() - start_time) / 60000))

    # save the full model!
    fs.save_deep_cnn_to_disk(cnn_model, model_name)
    workbook.close()
