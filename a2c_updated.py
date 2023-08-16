import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
from robogym.envs.rearrange.blocks_train import make_env
from tensorflow.python.keras.layers import Dense, Input , Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import adam_v2
import matplotlib.pyplot as plt
import os
import time

# Hyperparameters
LR_ACTOR = 0.01  # Learning rate for the actor network
LR_CRITIC = 0.01  # Learning rate for the critic network
GAMMA = 0.99  # Discount factor
NUM_EPISODES = 1000  # Number of episodes
MAX_STEPS = 50  # Maximum number of steps per episode
EPSILON = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0



# Define the Actor-Critic network
def build_actor_critic(state_dim, action_dim: np.ndarray):
    # Actor network
    input_state = Input(shape=(state_dim,))
    #input_state = Input(shape=state_dim)
    #flatten= Flatten()(input_state)
    dense1 = Dense(64, activation='relu')(input_state)
    dense2 = Dense(64, activation='relu')(dense1)
    output_actions=[]
    for i in range(action_dim.size):
            output_actions.append( Dense(action_dim[i],activation='softmax')(dense2))
    #output_actions = Dense(action_dim.sum(), activation='softmax')(dense2)
    # NOTE: you must not use 6 neurons in the output layer, because the action space is 6-dimensional.
    #       instead, you must use 66 neurons in the output layer, and group them into 6 groups of 11 neurons each
    #       then, you must apply a softmax activation function to each group of 11 neurons
    #       The best practice is to use 6 separate output layers with 11 neurons each, and apply a softmax activation
    #       function to each of them separately (why?).

    # Critic network
    dense3 = Dense(64, activation='relu')(input_state)
    dense4 = Dense(64, activation='relu')(dense3)
    output_value = Dense(6, activation='linear')(dense4)

    # Create the actor-critic model
    actor = Model(inputs=input_state, outputs=output_actions)
    critic = Model(inputs=input_state, outputs=output_value)

    # Compile the actor and critic models
    #actor.compile(optimizer=adam_v2.Adam(lr=LR_ACTOR), loss='categorical_crossentropy')
    #critic.compile(optimizer=adam_v2.Adam(lr=LR_CRITIC), loss='mean_squared_error')
    
    actor.summary()
    critic.summary()
    # NOTE: you can use summary() to print the model overall structure, and check if it is correct or not
    return actor, critic


# A2C agent
class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor, self.critic = build_actor_critic(state_dim, action_dim)
        self.loss= tf.keras.losses.Huber()
        self.actor_optimizer= tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)
        self.critic_optimizer= tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)
    def get_action(self, state):
        #state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state.flatten(),axis=0)
        # test= self.actor.predict(state)
        actor_output = self.actor(state)
        # the output of the model is list of 6 tensors in which all of them are in the shape (1,11)
        # action_probs =[]
        # for i in test:
        #     action_probs.append(i[0])
        #action_probs = self.actor.predict(state).flatten().tolist()
        # action = np.random.choice(self.action_dim, 1, p=action_probs)[0]
        # NOTE: as i described above, you must group the 66 neurons into 6 groups of 11 neurons each
        actions = []
        best_action_probs = []
        for arr in actor_output:
            #for i in self.action_dim:
                #tmp = arr[:i]
                #tmp = tmp / np.sum(tmp)
            # NOTE: do you remember I advised you to use 6 separate output layers with 11 neurons each? if you do that,
            #       you can change 2 recent lines of code
            
            arr_numpy = np.array(arr)[0] / np.sum(arr)
            action = np.random.choice(len(arr_numpy), 1, p=arr_numpy)[0]
            actions.append(action)
            best_action_probs.append(arr[0,action])

        critic_value= self.critic(state)
           
            #action_probs = arr[i:]
        return actions,best_action_probs,critic_value

    def train(self, states, actions, rewards, next_states, dones):
        states = np.expand_dims(np.array(states['obj_pos']).ravel(), axis=0)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        next_states= np.expand_dims(next_states.ravel(), axis=0)

        # Compute TD targets
        next_values = self.critic.predict(next_states)
        td_targets = rewards + GAMMA * next_values * (1 - dones)
        # NOTE: are you sure this is the correct formula for TD targets? I'm not sure about it

        # Train critic
        tmp= convert_to_tensor(states)
        td = convert_to_tensor(td_targets)
        self.critic.fit(states, td_targets, verbose=0)

        # Compute advantages
        values = self.critic.predict(states)
        advantages = td_targets - values
        #print(td_targets, values, rewards)

        # One-hot encode actions
        #print(actions)
        actions_onehot = np.eye(max(self.action_dim),11)[actions]
        # actions_onehot = np.eye(max(self.action_dim),self.action_dim.size)[actions]
        #print(actions_onehot)
        actions_onehot = np.expand_dims(actions_onehot, axis=0)
        for val in actions_onehot:
            for val2 in val:
                val2= np.expand_dims(val2, axis=0)
                actions_onehot=val2
       #print(advantages)
        
        # Train actor
        self.actor.fit(states, actions_onehot, sample_weight=advantages, verbose=0)

def reward_func(obj_pos, gripper_pos):
    distance = np.sqrt(np.sum((obj_pos-gripper_pos)**2))
    return 2**(-distance), distance


# Create the RoboGym environment
env = make_env(
    constants={
        'success_reward': 5.0,
        'success_pause_range_s': [0.0, 0.5],
        'max_timesteps_per_goal_per_obj': 600,
        'vision': True,  # use False if you don't want to use vision observations
        'vision_args': {
            'image_size': 200,
            'camera_names': ['vision_cam_front'],
            'mobile_camera_names': ['vision_cam_wrist'],
        },
        'goal_args': {
            'rot_dist_type': 'full',
            'randomize_goal_rot': True,
            'p_goal_hide_robot': 1.0,
        },
        'success_threshold': {'obj_pos': 0.04, 'obj_rot': 0.2},
    },
    parameters={
        'simulation_params': {
            'num_objects': 1,
            'max_num_objects': 1,
            'object_size': 0.0254,
            'used_table_portion': 1.0,
            'goal_distance_ratio': 1.0,
            'cast_shadows': False,
            'penalty': {
                # Penalty for collisions (wrist camera, table)
                'wrist_collision': 0.0,
                'table_collision': 0.0,

                # Penalty for safety stops
                'safety_stop_penalty': 0.0,
            }
        }
    }
)

# Get the state and action dimensions
state_dim = env.observation_space.spaces['obj_pos'].shape
# NOTE: are you sure you want to use 'goal_obj_pos' or 'obj_pos' as the state?
#       its shape is (32,3) and you are passing this tuple as the input_shape!
#       you may need to flatten this array and use 32*3=96 as the input_shape! I'm not sure about this!
#       on the other hand, are you sure using only the 'goal_obj_pos' as the state is enough? I think it's better to use
#       the robot's position or rotation as well
# action_dim = env.action_space.nvec.shape
action_dim = env.action_space.nvec
# NOTE: the action space is now a tuple of 6 elements, each element is the number of actions for each joint
#       now it is [11, 11, 11, 11, 11, 11]
# Create the A2C agent
running_reward = 0
agent = A2CAgent(state_dim[1], action_dim)
#episode_rewards = []
min_distances = []
max_distances = []
running_rewards = []
# Training loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    
    action_probs= []
    critic_values = []
    rewards_history = []
    distances = []
    with tf.GradientTape() as actor_tape:
        with tf.GradientTape() as critic_tape:
            for step in range(MAX_STEPS):
                # Get an action from the agent

                action,action_prob,critic_value = agent.get_action(state['obj_pos'])
                
                # Take a step in the environment
                next_state, reward, done, _ = env.step(action)

                reward, distance = reward_func(state['obj_pos'],state['gripper_pos'])
                action_probs.append(tf.math.log(action_prob))
                critic_values.append(critic_value)
                rewards_history.append(reward)
                distances.append(distance)

                #print(next_state.keys())
                #print(next_state['obj_pos'])
                #print(next_state['gripper_pos'])

                # Store the transition in memory
                #agent.train(state, action, reward, next_state['obj_pos'], done)

                #agent.memory.append((state, action, reward, next_state['obj_pos'], done))
                # NOTE: this is an obvious error :) good luck fixing it...

                # Update the current state and episode reward
                state = next_state
                episode_reward += reward
                #episode_rewards.append(episode_reward)
                if done:
                    # Print the episode reward
                    print(f"Episode: {episode + 1}, Reward: {episode_reward}")
                    break
            # input("press enter")
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + GAMMA * discounted_sum
                returns.insert(0, discounted_sum)
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + EPSILON)
            returns = returns.tolist()
            history = zip(action_probs, critic_values, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    agent.loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # input("rrrrrr")
            grads = actor_tape.gradient(sum(actor_losses), agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))
            grads = critic_tape.gradient(sum(critic_losses), agent.critic.trainable_variables)
            agent.critic_optimizer.apply_gradients(zip(grads, agent.critic.trainable_variables))
            # input("eeeeeee")
            action_probs.clear()
            critic_values.clear()
            rewards_history.clear()

        if episode % 1 == 0:
            template = "running reward: {:.2f} at episode {}, \n min distance: {}"
            min_distance = min(distances)
            max_distance = max(distances)
            print(template.format(running_reward, episode,min_distance))
            running_rewards.append(running_reward)
            min_distances.append(min_distance)
            max_distances.append(max_distance)

            

    # if running_reward > 195:  # Condition to consider the task solved
    #     print("Solved at episode {}!".format(episode))
    #     break

folder_name = str(time.time())
os.mkdir(folder_name)

plt.plot(running_rewards, label="running rewrds")
plt.legend()
plt.savefig(os.path.join(folder_name, "running reward.png"))

plt.figure()
plt.plot(min_distances, "-o", label="min distances")
plt.plot(max_distances, "-o", label="max distances")
for i, (min_dist, max_dist) in enumerate(zip(min_distances, max_distances)):
    plt.plot((i,i), (min_dist, max_dist), "--", color="#444444")
plt.legend()
plt.savefig(os.path.join(folder_name, "min dis.png"))

with open(os.path.join(folder_name, "hyperparameters.txt"), 'w') as f:
        f.write("LR_ACTOR: "+ str(LR_ACTOR))
        f.write('\n')
        f.write("LR_CRITIC: "+str(LR_CRITIC))
        f.write('\n')
        f.write("GAMMA: "+str(GAMMA))
        f.write('\n')
        f.write("NUM_EPISODES: "+str(NUM_EPISODES))
        f.write('\n')
        f.write("MAX_STEPS: "+str(MAX_STEPS))
        f.write('\n')
