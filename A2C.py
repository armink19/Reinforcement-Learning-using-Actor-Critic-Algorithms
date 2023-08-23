import numpy as np
import tensorflow as tf
from robogym.envs.rearrange.blocks_train import make_env
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import os
import time

# Hyperparameters
LR_ACTOR = 0.01  # Learning rate for the actor network
LR_CRITIC = 0.01  # Learning rate for the critic network
GAMMA = 0.99  # Discount factor
NUM_EPISODES = 1000  # Number of episodes
MAX_STEPS = 100 # Maximum number of steps per episode
EPSILON = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0



# Define the Actor-Critic network
def build_actor_critic(state_dim, action_dim: np.ndarray):
    # Actor network
    input_state = Input(shape=(state_dim,))
    dense1 = Dense(64, activation='relu')(input_state)
    dense2 = Dense(64, activation='relu')(dense1)
    output_actions=[]
    for i in range(action_dim.size):
            output_actions.append( Dense(action_dim[i],activation='softmax')(dense2))
    
    # Critic network
    dense3 = Dense(64, activation='relu')(input_state)
    dense4 = Dense(64, activation='relu')(dense3)
    output_value = Dense(6, activation='linear')(dense4)

    # Create the actor-critic model
    actor = Model(inputs=input_state, outputs=output_actions)
    critic = Model(inputs=input_state, outputs=output_value)

    
    actor.summary()
    critic.summary()
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
        obj_pos= state['obj_pos']
        gripper_pos= state['gripper_pos']
        gripper_pos= np.expand_dims(gripper_pos,axis=0)
        target_pos= state['goal_obj_pos']
        input_array = np.concatenate([gripper_pos, obj_pos,target_pos], axis=0)
        state_tensor = tf.convert_to_tensor(input_array, dtype=tf.float64)
        
        actor_output = self.actor(state_tensor)
       
        actions = []
        best_action_probs = []
        for arr in actor_output:
            arr_numpy = np.array(arr)[0]
            arr_sum = np.sum(arr_numpy)
            if arr_sum != 0:
                arr_prob = arr_numpy / arr_sum
                action = np.random.choice(len(arr_prob), 1, p=arr_prob)[0]
                actions.append(action)
                best_action_probs.append(arr[0, action])
            else:
                # Handle the case where all probabilities are zero , random selection in this case
                action = np.random.choice(len(arr_numpy), 1)[0]
                actions.append(action)
                best_action_probs.append(arr[0, action])

        critic_value= self.critic(state_tensor)
           
        return actions,best_action_probs,critic_value


def reward_func(obj_pos, gripper_pos):
    distance = np.sqrt(np.sum((obj_pos-gripper_pos)**2))
    return 2**(-distance), distance


# Create the RoboGym environment
env = make_env(
    constants={
        'success_reward': 5.0,
        'success_pause_range_s': [0.0, 0.5],
        'max_timesteps_per_goal_per_obj': 600,
        'vision': False,  # use False if you don't want to use vision observations
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
            'max_num_objects': 8,
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
action_dim = env.action_space.nvec

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

                action,action_prob,critic_value = agent.get_action(state)
                
                # Take a step in the environment
                next_state, _, _, _ = env.step(action)

                reward, distance = reward_func(state['obj_pos'],state['gripper_pos'])
                action_probs.append(tf.math.log(action_prob))
                critic_values.append(critic_value)
                rewards_history.append(reward)
                distances.append(distance)

                # Update the current state and episode reward
                state = next_state
                episode_reward += reward
                
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
                advantage = ret - value
                actor_losses.append(-log_prob * advantage)  
                critic_losses.append(
                    agent.loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            grads_actor = actor_tape.gradient(sum(actor_losses), agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(grads_actor, agent.actor.trainable_variables))
            grads_critic = critic_tape.gradient(sum(critic_losses), agent.critic.trainable_variables)
            agent.critic_optimizer.apply_gradients(zip(grads_critic, agent.critic.trainable_variables))
            action_probs.clear()
            critic_values.clear()
            rewards_history.clear()

            template = "running reward: {:.2f} at episode {}, \n min distance: {}"
            min_distance = min(distances)
            max_distance = max(distances)
            print(template.format(running_reward, episode,min_distance))
            running_rewards.append(running_reward)
            min_distances.append(min_distance)
            max_distances.append(max_distance)



folder_name = str(time.time())
os.mkdir(folder_name)
plt.style.use('seaborn')
plt.plot(running_rewards, label='Running Reward', color='blue', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('A2C Algorithm Performance')
plt.legend()
plt.savefig(os.path.join(folder_name, "plot.png"))

with open(os.path.join(folder_name, "hyperparameters.txt"), 'w') as f:
        f.write("Basic A2C")
        f.write('\n')
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
        f.write("Reward: "+str(running_reward))
