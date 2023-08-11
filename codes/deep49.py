import gym
env = gym.make('CartPole-v0')
# print(env.action_space.)
# #> Discrete(2)
# print(env.observation_space)
# #> Box(4,)
observation = env.reset()
actions = [0,0,1,1,1]
for t in range(1000):

    env.render()

    cart_pos , cart_vel , pole_ang , ang_vel = observation

    # Move Cart Right if Pole is Falling to the Right

    action = actions[t%5]
    # Perform Action
    observation , reward, done, info = env.step(action)
    print(reward)
