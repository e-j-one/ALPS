from typing import Dict, List, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import gymnasium as gym

from params import TrainArgs
from envs import RoomEnv
from abc import ABC, abstractmethod


def normalize_obs(obs: jnp.ndarray) -> jnp.ndarray:
    """normalize observation if it's an image"""
    if obs.dtype == jnp.uint8:
        return obs.astype(jnp.float32) / 255.0
    return obs


# dynamics training utils
def sample_sequence_batch(rng: np.random.Generator, all_observations: np.ndarray, all_actions: np.ndarray, 
                              valid_ep_starts: np.ndarray, valid_ep_lens: np.ndarray, batch_size: int, horizon: int
                              ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """samples sequences for training autoregressive dynamics model"""
    num_valid = len(valid_ep_starts)
    sampled_indices = rng.integers(0, num_valid, size=batch_size)

    starts = valid_ep_starts[sampled_indices]
    lengths = valid_ep_lens[sampled_indices]

    # select random offsets within episode lengths
    max_offset = lengths - horizon - 1
    max_offset = np.maximum(max_offset, 0)
    offsets = np.floor(rng.random(batch_size) * (max_offset + 1)).astype(np.int32)
    current_indices = starts + offsets
    
    # extract start observations
    batch_start_obs = all_observations[current_indices]  # (batch, obs_dim)

    # extract action sequences [t ... t+H-1]
    seq_shifts = np.arange(horizon)
    act_idx = current_indices[:, None] + seq_shifts[None, :]
    batch_actions = all_actions[act_idx]  # (batch, H, action_dim)

    # extract target sequences [t+1 ... t+H]
    target_idx = current_indices[:, None] + seq_shifts[None, :] + 1
    batch_target_obs = all_observations[target_idx]  # (batch, H, obs_dim)

    # transfer to gpu and normalize
    return (
        normalize_obs(jax.device_put(batch_start_obs)),
        jax.device_put(batch_actions),
        normalize_obs(jax.device_put(batch_target_obs))
    )


# prior training utils
def sample_prior_batch(rng: np.random.Generator, all_observations: np.ndarray, all_z: np.ndarray, all_actions: np.ndarray, 
                           valid_ep_starts: np.ndarray, valid_ep_lens: np.ndarray, batch_size: int, min_horizon: int, max_horizon: int
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """samples (current_obs, current_z, future_z, expert_action) tuples for training the prior model with variable horizons"""
    num_valid = len(valid_ep_starts)
    sampled_indices = rng.integers(0, num_valid, size=batch_size)

    starts = valid_ep_starts[sampled_indices]
    lengths = valid_ep_lens[sampled_indices]

    # sample random horizon for each sample in the batch (uniform in [min_horizon, max_horizon])
    horizons = rng.integers(min_horizon, max_horizon + 1, size=batch_size)
    horizons = np.minimum(horizons, lengths - 1)  # ensure horizon is less than episode length

    # sample time offsets within each episode
    max_offsets = lengths - horizons
    max_offsets = np.maximum(max_offsets, 0)
    offsets = np.floor(rng.random(batch_size) * max_offsets).astype(np.int32)

    idx_t = starts + offsets
    idx_future = idx_t + horizons

    # gather data
    current_obs = all_observations[idx_t]
    current_z = all_z[idx_t]
    future_z = all_z[idx_future]
    expert_action = all_actions[idx_t]
    
    # transfer to gpu and normalize
    return (
        normalize_obs(jax.device_put(current_obs)),
        jax.device_put(current_z),
        jax.device_put(future_z),
        jax.device_put(expert_action)
    )


class Buffer(ABC):
    """replay buffer for continuous state and action spaces"""
    def __init__(self, args: TrainArgs, capacity: int, obs_shape: tuple, action_dim: int, key: jax.random.PRNGKey = None, allocate: bool = True):
        self.args = args
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # initialize storage arrays (filled in by add_episode or load_offline_dataset)
        self.observations = None
        self.actions = None
        self.time_to_end = None
        self.episode_ids = None
        self.valid_indices = None
        self.valid_count = 0

        if allocate:
            self._allocate(capacity)

        # metadata
        self.current_size = 0
        self.episode_lengths = []
        self.episode_start_indices = []
        self.num_episodes = 0

        # cached numpy arrays
        self._ep_starts_np = None
        self._ep_lens_np = None

        # shard rotation for sharded (100M/1B) datasets; inactive unless enable_shard_rotation is called
        self.shard_paths = []
        self.shard_idx = 0
        self._shard_interval = 0
        self._shard_loader = None

        seed = int(jax.random.randint(key, (1,), 0, 1000000)[0]) if key is not None else args.seed
        self.rng = np.random.default_rng(seed)

    def _allocate(self, capacity: int):
        """allocate storage arrays"""
        self.capacity = capacity
        if len(self.obs_shape) == 3:
            self.observations = np.zeros((capacity,) + self.obs_shape, dtype=np.uint8)
        else:
            self.observations = np.zeros((capacity,) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.time_to_end = np.zeros(capacity, dtype=np.int32)
        self.episode_ids = np.zeros(capacity, dtype=np.int32)
        self.valid_indices = np.zeros(capacity, dtype=np.int32)

    def enable_shard_rotation(self, shard_paths: List[str], interval: int, loader):
        """rotate through dataset shards, holding one in memory at a time; shard 0 must already be loaded"""
        self.shard_paths = list(shard_paths)
        self.shard_idx = 0
        self._shard_interval = interval
        self._shard_loader = loader

    def maybe_rotate_shard(self, num_updates: int) -> bool:
        """load the next shard every `interval` updates (as in horizon-reduction); returns True if the data changed"""
        if len(self.shard_paths) <= 1 or self._shard_interval <= 0 or num_updates % self._shard_interval != 0:
            return False
        self._load_shard((self.shard_idx + 1) % len(self.shard_paths))
        return True

    def reset_shard(self):
        """reload the first shard so evaluation always uses the same data"""
        if self.shard_paths and self.shard_idx != 0:
            self._load_shard(0)

    def _load_shard(self, idx: int):
        self.shard_idx = idx
        self.load_offline_dataset(self._shard_loader(self.shard_paths[idx]))
        tqdm.write(f"  Loaded shard {idx + 1}/{len(self.shard_paths)}")

    def _cache_episode_arrays(self):
        """cache episode metadata"""
        self._ep_starts_np = np.array(self.episode_start_indices)
        self._ep_lens_np = np.array(self.episode_lengths)
    
    def add_episode(self, episode_data: Dict[str, List]):
        """add episode to buffer"""
        if 'observations' not in episode_data:
            raise ValueError("Episode must contain 'observations'")

        observations = np.array(episode_data['observations'])

        if self.observations.dtype == np.uint8:
            if self.args.env_type == 'RoomEnv' and observations.dtype != np.uint8:
                # scale 0.0-1.0 to 0-255
                observations = (observations * 255).astype(np.uint8)
            else:
                # observations are already in 0-255 range
                observations = observations.astype(np.uint8)
        else:
            observations = observations.astype(np.float32)

        episode_length = len(observations)

        if episode_length < 2:
            return False  # need at least 2 observations to form a transition

        # check capacity
        if self.current_size + episode_length > self.capacity:
            return False
        
        # get indices where this episode will be stored
        start_idx = self.current_size
        end_idx = start_idx + episode_length
        
        self.episode_start_indices.append(start_idx)
        
        # prepare actions
        raw_actions = np.array(episode_data['actions'], dtype=np.float32)

        # if actions are scalars (T,), reshape them to (T, 1)
        if raw_actions.ndim == 1:
            raw_actions = raw_actions[..., None]

        # handle action padding
        if len(raw_actions) == len(observations) - 1:
            # pad one dummy action for the terminal observation so shapes match
            padding = np.zeros((1, raw_actions.shape[1]), dtype=np.float32)
            episode_actions = np.concatenate([raw_actions, padding], axis=0)
        elif len(raw_actions) == len(observations):
            # action for terminal observation is usually dummy/zero
            episode_actions = raw_actions
        else:
            # fallback
            episode_actions = raw_actions
        
        # prepare metadata
        time_values = np.arange(episode_length)[::-1].astype(np.int32)
        
        # store in main buffer
        self.observations[start_idx:end_idx] = observations
        self.actions[start_idx:end_idx] = episode_actions
        self.time_to_end[start_idx:end_idx] = time_values
        self.episode_ids[start_idx:end_idx] = self.num_episodes

        new_valid_count = episode_length - 1
        if new_valid_count > 0:
            # generate indices: start_idx, start_idx+1, ..., end_idx-2
            new_indices = np.arange(start_idx, end_idx - 1, dtype=np.int32)
            
            # append to valid_indices buffer
            v_start = self.valid_count
            v_end = v_start + new_valid_count
            self.valid_indices[v_start:v_end] = new_indices
            self.valid_count = v_end

        self.current_size = end_idx
        self.num_episodes += 1
        self.episode_lengths.append(episode_length)
        return True

    def load_offline_dataset(self, dataset: Dict[str, np.ndarray]):
        """
        load an offline dataset (e.g., from ogbench/D4RL) into the buffer
        dataset is dictionary which can be compact (with 'valids') or standard (with 'next_observations' and 'terminals')
        """
        if 'valids' in dataset:
            observations = dataset['observations']
            actions = dataset['actions']
            valids = dataset['valids'].astype(bool)
            N = len(observations)

            if len(self.obs_shape) == 3:
                if observations.dtype == np.uint8:
                    self.observations = observations
                else:
                    self.observations = observations.astype(np.uint8)
            else:
                if observations.dtype == np.float32:
                    self.observations = observations
                else:
                    self.observations = observations.astype(np.float32)

            # inject actions
            if actions.ndim == 1:
                actions = actions[:, None]
            if actions.dtype == np.float32:
                self.actions = actions
            else:
                self.actions = actions.astype(np.float32)

            # compute episode boundaries from valids mask
            end_indices = np.where(~valids)[0]
            if len(end_indices) == 0 or end_indices[-1] != N - 1:
                end_indices = np.append(end_indices, N - 1)
            start_indices = np.concatenate([[0], end_indices[:-1] + 1])
            episode_lengths = end_indices - start_indices + 1

            # compute time to end for each transition
            positions = np.arange(N)
            episode_ids_map = np.searchsorted(end_indices, positions, side='left')
            self.time_to_end = (end_indices[episode_ids_map] - positions).astype(np.int32)
            self.episode_ids = episode_ids_map.astype(np.int32)

            valid_indices_array = np.where(valids)[0].astype(np.int32)
            self.valid_count = len(valid_indices_array)
            self.valid_indices = valid_indices_array

            # update metadata
            self.capacity = N
            self.current_size = N
            self.episode_start_indices = start_indices.tolist()
            self.episode_lengths = episode_lengths.tolist()
            self.num_episodes = len(start_indices)
        else:
            observations = dataset['observations']
            actions = dataset['actions']

            has_next = 'next_observations' in dataset
            if has_next:
                next_observations = dataset['next_observations']

            # determine episode ends
            terminals = dataset['terminals'].astype(bool)
            end_indices = np.where(terminals)[0]

            if len(end_indices) == 0 or end_indices[-1] != len(observations) - 1:
                end_indices = np.append(end_indices, len(observations) - 1)

            # allocate if not already done
            if self.observations is None:
                N = len(observations)
                num_episodes = len(end_indices)
                capacity = N + num_episodes if has_next else N
                self._allocate(capacity)

            start_idx = 0
            for end_idx in tqdm(end_indices, desc="Processing Trajectories"):
                # get the main sequence: o0...oT-1
                curr_obs = observations[start_idx : end_idx + 1]
                curr_acts = actions[start_idx : end_idx + 1]

                if has_next:
                    # append final next observation
                    final_next_obs = next_observations[end_idx]
                    full_obs = np.concatenate([curr_obs, final_next_obs[None]], axis=0)
                else:
                    full_obs = curr_obs

                if self.current_size + len(full_obs) > self.capacity:
                    print(f"Buffer full at {self.current_size}. Stopping load.")
                    break

                self.add_episode({'observations': full_obs, 'actions': curr_acts})
                start_idx = end_idx + 1

        self._cache_episode_arrays()
    
    @abstractmethod
    def sample(self, batch_size: int, **kwargs) -> Tuple:
        """sampling method for ALLO training"""
        raise NotImplementedError
    
    @abstractmethod
    def sample_uniform_batch(self, batch_size: int, **kwargs) -> jnp.ndarray:
        """sample observations uniformly from all observations"""
        raise NotImplementedError

    def get_all_observations(self) -> np.ndarray:
        """get all observations currently in buffer"""
        return self.observations[:self.current_size]

    def get_episode(self, episode_idx: int) -> Dict[str, jnp.ndarray]:
        """get a specific episode by index"""
        start_idx = self.episode_start_indices[episode_idx]
        length = self.episode_lengths[episode_idx]
        end_idx = start_idx + length

        obs_cpu = self.observations[start_idx:end_idx]
        act_cpu = self.actions[start_idx:end_idx-1]
        
        # transfer to GPU and normalize
        return {
            'observations': normalize_obs(jax.device_put(obs_cpu)),
            'actions': jax.device_put(act_cpu),
        }
    
    def get_total_transitions(self) -> int:
        """get total number of valid transitions in buffer"""
        return sum(length - 1 for length in self.episode_lengths)

    def generate_continuous_buffer(self, env: Union[RoomEnv, gym.Env]):
        """generate buffer through random exploration"""
        obs, _ = env.reset()
        episode_obs = [obs]
        episode_acts = []
        
        # calculate how much space is left
        space_left = self.capacity - self.current_size
        pbar = tqdm(total=space_left, desc="Generating Buffer", unit=" steps")
        
        while self.current_size < self.capacity:
            # select random action
            action = env.action_space.sample()
            
            next_obs, _, _, truncated, _ = env.step(action)
            
            # store data in episode lists
            episode_obs.append(next_obs)
            episode_acts.append(action)
            
            done = truncated    # just collect episode until truncation
            
            # load episode if done into buffer
            if done:
                added = self.add_episode({'observations': episode_obs, 'actions': episode_acts})
                if added:
                    pbar.update(len(episode_obs))
                else:
                    break

                # reset environment
                obs, _ = env.reset()
                episode_obs = [obs]
                episode_acts = []
            else:
                obs = next_obs

        pbar.close()
        self._cache_episode_arrays()
        

class DiscountedReplayBuffer(Buffer):
    """buffer that samples future observations using a discounted geometric distribution"""
    def sample(self, batch_size: int, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """sample (current, next, random) transitions"""
        discount = kwargs.get('discount')
        if discount is None:
            raise ValueError("sample() requires a 'discount' keyword argument.")
        rng = kwargs.get('rng', self.rng)

        if self.valid_count == 0:
            raise ValueError("Buffer has no valid transitions")
        
        # sample valid start indices
        batch_idx_positions = rng.integers(0, self.valid_count, size=batch_size)
        current_indices = self.valid_indices[batch_idx_positions]

        # sample next indices with discounted geometric distribution over time to end
        ranges = self.time_to_end[current_indices] - 1
        ranges = np.maximum(ranges, 0)

        if discount == 0.0:
            dist = np.zeros_like(ranges)
        elif discount == 1.0:
            seeds = rng.random(size=batch_size)
            dist = np.floor(seeds * ranges).astype(np.int32)
        else:
            seeds = rng.random(size=batch_size)
            # inverse transform sampling: k = log(1 - u(1 - gamma^N)) / log(gamma)
            dist = np.floor(np.log(1 - (1 - np.power(discount, ranges)) * seeds) / np.log(discount))
            dist = dist.astype(np.int32)

        next_indices = current_indices + dist + 1

        # sample random indices uniformly from the same episodes
        episode_ids = self.episode_ids[current_indices]
        ep_start = self._ep_starts_np[episode_ids]
        ep_len = self._ep_lens_np[episode_ids]
        random_offsets = rng.integers(0, ep_len.max(), size=batch_size) % ep_len    # assuming all episodes are of same length, this will just be uniform random sampling within the episode
        random_indices = ep_start + random_offsets

        # gather data
        batch_curr = self.observations[current_indices]
        batch_next = self.observations[next_indices]
        batch_rand = self.observations[random_indices]

        # transfer to gpu and normalize
        curr_jax = normalize_obs(jax.device_put(batch_curr))
        next_jax = normalize_obs(jax.device_put(batch_next))
        rand_jax = normalize_obs(jax.device_put(batch_rand))

        return curr_jax, next_jax, rand_jax

    def sample_uniform_batch(self, batch_size: int, **kwargs) -> jnp.ndarray:
        """sample observations uniformly from all observations"""
        rng = kwargs.get('rng', self.rng)
        if self.current_size == 0:
            raise ValueError("Buffer is empty")

        indices = rng.integers(0, self.current_size, size=batch_size)
        batch_obs = self.observations[indices]

        return normalize_obs(jax.device_put(batch_obs))

    def sample_standard_transitions(self, batch_size: int, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """sample single-step transitions"""
        c, n, _ = self.sample(batch_size, discount=0.0, **kwargs)
        return c, n