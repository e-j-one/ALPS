import os
import ogbench
import imageio

task = 'pointmaze-medium-navigate-v0'
env = ogbench.make_env_and_datasets(task, env_only=True)
obs, info = env.reset()

frames = []
frames.append(env.render())

done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frames.append(env.render())
    step += 1

# save video
os.makedirs(f"./viz/{task}", exist_ok=True)
imageio.mimsave(f"./viz/{task}/episode.mp4", frames, fps=30)
env.close()
