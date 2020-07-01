
def run():
  import numpy as np
  import srg

  env = srg.make('Pendulum-v0')

  for i_episode in range(20):
    state = env.reset()
    # integral = 0
    # derivative = 0
    # prev_error = 0
    for t in range(500):
      env.render()
      state, reward, done, info = env.step([1])
      if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

  env.close()


if __name__ == '__main__':
  run()