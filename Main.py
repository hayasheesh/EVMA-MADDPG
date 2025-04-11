import numpy as np
from Config import NUM_EPISODES, STATE_DIM, ACTION_DIM, NUM_AGENTS, EPISODE_STEPS, INITIAL_SOC, CAPACITY, AG_REQUEST
from EVEnv import EVEnv
from Agent import MADDPG
from Utils import plot_episode_data, plot_episode_rewards

def train_maddpg_only(num_episodes=NUM_EPISODES, use_transformer=False):
    env = EVEnv(capacity=CAPACITY, initial_soc=INITIAL_SOC, ag_request=AG_REQUEST, 
               episode_steps=EPISODE_STEPS)
    agent = MADDPG(STATE_DIM, ACTION_DIM, num_agents=NUM_AGENTS, use_transformer=use_transformer)
    
    episode_rewards = []
    save_folder = r"C:\Users\hayas\Desktop\EVMA_Local\結果保存"
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(env.episode_steps):
            actions = agent.get_actions(state, add_noise=True)
            
            next_state, rewards, done, _ = env.step(actions)
            
            agent.buffer.cache(state, next_state, actions, rewards, done)
            
            state = next_state
            episode_reward += np.sum(rewards)
            
            if np.all(done):
                break
                
            if (t + 1) % 1 == 0:
                agent.update()
        
        episode_rewards.append(episode_reward)
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}: Average Reward: {avg_reward:.3f}")
            
        if (ep + 1) % 1000 == 0:
            test_rewards = []
            print(f"Running test mode at episode {ep+1}...")
            for test_run in range(10):
                test_reward = test_maddpg_only(agent, env, save_folder, ep+1, num_test_episodes=1, test_idx=test_run)
                test_rewards.append(test_reward)
            avg_test_reward = np.mean(test_rewards)
            print(f"Average test reward over 10 runs: {avg_test_reward:.3f}")
    
    return agent, episode_rewards

def test_maddpg_only(agent, env, save_folder, current_episode, num_test_episodes=1, test_idx=0):
    total_rewards = []
    
    for ep in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        ag_requests_steps = []
        ev1_actions = []
        ev2_actions = []
        ev3_actions = []
        
        initial_soc = {
            'ev1': env.soc["ev1"],
            'ev2': env.soc["ev2"],
            'ev3': env.soc["ev3"]
        }
        
        for t in range(env.episode_steps):
            current_request = env.ag_request
            ag_requests_steps.append(current_request)
            
            actions = agent.get_actions(state, add_noise=False)
            
            ev1_actions.append(actions[0][0])
            ev2_actions.append(actions[1][0])
            ev3_actions.append(actions[2][0])
            
            next_state, rewards, done, info = env.step(actions)
            
            state = next_state
            episode_reward += np.sum(rewards)
            
            if np.all(done):
                break
        
        total_rewards.append(episode_reward)
        
        min_len = min(len(ag_requests_steps), len(ev1_actions), len(ev2_actions), len(ev3_actions))
        ag_requests_steps = ag_requests_steps[:min_len]
        ev1_actions = ev1_actions[:min_len]
        ev2_actions = ev2_actions[:min_len]
        ev3_actions = ev3_actions[:min_len]
        
        plot_data = {
            'actual_ev1': ev1_actions,
            'actual_ev2': ev2_actions,
            'actual_ev3': ev3_actions,
            'ag_requests': ag_requests_steps
        }
        plot_episode_data(
            plot_data, 
            f"{save_folder}/maddpg_only_episode_{current_episode}_test{test_idx+1}.png", 
            episode_num=current_episode,
            initial_soc=initial_soc
        )
    
    avg_reward = np.mean(total_rewards)
    print(f"Test {test_idx+1} at Episode {current_episode}: Reward: {avg_reward:.3f}")
    
    return avg_reward

if __name__ == '__main__':
    # 通常のMADDPGを使用する場合
    # train_maddpg_only()
    
    # TransformerベースのMADDPGを使用する場合
    train_maddpg_only(use_transformer=True)
