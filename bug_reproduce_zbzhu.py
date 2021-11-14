import numpy as np
import time
import gym
import ctypes
import inspect
import sys
from gym import Env
from collections import defaultdict
from typing import Any, List, Tuple, Optional, Union, Callable, Dict
from dataclasses import replace
from collections import OrderedDict
from multiprocessing.context import Process
from multiprocessing import Array, Pipe, connection

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider

from smarts_imitation.utils import agent
import cloudpickle


def list_dict_to_dict_list(list_dict):
    dict_list = defaultdict(list)
    for _dict in list_dict:
        for k, v in _dict.items():
            dict_list[k].append(v)
    return dict(dict_list)


def dict_list_to_list_dict(dict_list):
    # For example,
    # Input: {"agent_0": [1, 2], "agent_1": [3, 4]}
    # Output: [{"agent_0": 1, "agent_1": 3}, {"agent_0": 2, "agent_1": 4}]
    list_dict = [
        {k: v[idx] for k, v in dict_list.items()}
        for idx in range(len(list(dict_list.values())[0]))
    ]
    return list_dict


class RunningMeanStd(object):
    """Calulates the running mean and std of a data stream.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, mean=0.0, std=1.0) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


class SMARTSImitation(gym.Env):
    # def __init__(self, scenarios, action_range):
    def __init__(self, scenarios):
        super(SMARTSImitation, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.agent_spec = agent.get_agent_spec(self.obs_stacked_size)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64
        )

        # assert (
        #     action_range.shape == (2, 2) and (action_range[1] >= action_range[0]).all()
        # ), action_range
        # self._action_range = action_range  # np.array([[low], [high]])

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=None,
        )
        print("SMARTSImitaion Initialized")

    def seed(self, seed):
        np.random.seed(seed)

    def _convert_obs(self, raw_observations):
        observation = self.agent_spec.observation_adapter(
            raw_observations[self.vehicle_id]
        )
        ego_state = []
        other_info = []
        for feat in observation:
            if feat in ["ego_pos", "speed", "heading"]:
                ego_state.append(observation[feat])
            else:
                other_info.append(observation[feat])
        ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
        other_info = np.concatenate(other_info, axis=1).reshape(-1)
        full_obs = np.concatenate((ego_state, other_info))
        full_obs = {"agent_0": full_obs}
        return full_obs

    def step(self, action):
        action = action["agent_0"]
        action = np.clip(action, -1, 1)
        # Transform the normalized action back to the original range
        # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
        # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
        # action = (self._action_range[1] - self._action_range[0]) * (
        #     action + 1
        # ) / 2 + self._action_range[0]

        raw_observations, rewards, dones, _ = self.smarts.step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)}
        )
        full_obs = self._convert_obs(raw_observations)

        info = {}
        info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
        info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0

        return (
            full_obs,
            {"agent_0": rewards[self.vehicle_id]},
            {"agent_0": dones[self.vehicle_id]},
            info,
        )

    def reset(self):
        if self.vehicle_itr >= len(self.vehicle_ids):
            self._next_scenario()

        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts.switch_ego_agents({self.vehicle_id: self.agent_spec.interface})

        observations = self.smarts.reset(self.scenario)
        full_obs = self._convert_obs(observations)
        self.vehicle_itr += 1
        return full_obs

    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_ids = list(self.vehicle_missions.keys())
        np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()


_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class ShArray:
    """Wrapper of multiprocessing Array."""

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                p.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                p.send(env.seed(data) if hasattr(env, "seed") else None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocEnvWorker:
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        self._env_fn = env_fn

    def __getattr__(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]]
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    def reset(self) -> Any:
        self.parent_remote.send(["reset", None])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send_action(self, action: np.ndarray) -> None:
        self.parent_remote.send(["step", action])

    def get_result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs, rew, done, info = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs, rew, done, info

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        return self.parent_remote.recv()

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()


class BaseVectorEnv(Env):
    """Base class for vectorized environments wrapper.
    Usage:
    ::
        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num
    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.
    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::
        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    .. warning::
        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::
            def seed(self, seed):
                np.random.seed(seed)
        Otherwise, the outputs of these envs may be the same with each other.
    :param env_fns: a list of callable envs, ``env_fns[i]()`` generates the ith env.
    :param worker_fn: a callable worker, ``worker_fn(env_fns[i])`` generates a
        worker which contains the i-th env.
    :param int wait_num: use in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.
    :param float timeout: use in asynchronous simulation same as above, in each
        vectorized step it only deal with those environments spending time
        within ``timeout`` seconds.
    :param bool norm_obs: Whether to track mean/std of data and normalise observation
        on return. For now, observation normalization only support observation of
        type np.ndarray.
    :param obs_rms: class to track mean&std of observation. If not given, it will
        initialize a new one. Usually in envs that is used to evaluate algorithm,
        obs_rms should be passed in. Default to None.
    :param bool update_obs_rms: Whether to update obs_rms. Default to True.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], SubprocEnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
        norm_obs: bool = False,
        obs_rms_n: Optional[Dict[str, RunningMeanStd]] = None,
        update_obs_rms: bool = True,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of SubprocEnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, SubprocEnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])
        self.agent_ids = self.workers[0].agent_ids
        self.n_agents = self.workers[0].n_agents

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert (
            1 <= self.wait_num <= len(env_fns)
        ), f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        self.timeout = timeout
        assert (
            self.timeout is None or self.timeout > 0
        ), f"timeout is {timeout}, it should be positive if provided!"
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[SubprocEnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

        # initialize observation running mean/std
        self.norm_obs = norm_obs
        self.update_obs_rms = update_obs_rms
        # by default, use distinct rms for different agents
        self.obs_rms_n = (
            dict(zip(self.agent_ids, [RunningMeanStd() for _ in range(self.n_agents)]))
            if obs_rms_n is None and norm_obs
            else obs_rms_n
        )
        self.__eps = np.finfo(np.float32).eps.item()

    def _assert_is_not_closed(self) -> None:
        assert (
            not self.is_closed
        ), f"Methods of {self.__class__.__name__} cannot be called after close."

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.
        Any class who inherits ``Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in [
            "metadata",
            "reward_range",
            "spec",
            "action_space",
            "observation_space",
        ]:  # reserved keys in Env
            return self.__getattr__(key)
        else:
            return super().__getattribute__(key)

    def __getattr__(self, key: str) -> List[Any]:
        """Fetch a list of env attributes.
        This function tries to retrieve an attribute from each individual wrapped
        environment, if it does not belong to the wrapping vector environment class.
        """
        return [getattr(worker, key) for worker in self.workers]

    def _wrap_id(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert (
                i not in self.waiting_id
            ), f"Cannot interact with environment {i} which is stepping now."
            assert (
                i in self.ready_id
            ), f"Can only interact with ready environments {self.ready_id}."

    def reset(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        obs_list = [self.workers[i].reset() for i in id]  # list of dict
        obs_dict = list_dict_to_dict_list(obs_list)  # dict of list
        for a_id in obs_dict.keys():
            try:
                obs_dict[a_id] = np.stack(obs_dict[a_id])
            except ValueError:  # different len(obs)
                obs_dict[a_id] = np.array(obs_dict[a_id], dtype=object)
            if self.obs_rms_n and self.update_obs_rms:
                self.obs_rms_n[a_id].update(obs_dict[a_id])
        # in case different agents share the same noramlizer, one should normlize obs
        # after all normalizers are updated.
        for a_id in obs_dict.keys():
            obs_dict[a_id] = self.normalize_obs(obs_dict[a_id], a_id)
        return dict(obs_dict)

    def step(
        self,
        action_n: Dict[str, np.ndarray],
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """Run one timestep of some environments' dynamics.
        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id, either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.
        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.
        :param numpy.ndarray action: a batch of action provided by the agent.
        :return: A tuple including four items:
            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        For the async simulation:
        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        action_n["agent_0"] = action_n["agent_0"][: len(id)]
        print("agent_ids:{}".format(self.agent_ids))
        # print("len(id):{} action:{}".format(len(id),action_n["agent_0"]))
        if not self.is_async:
            for action in action_n.values():
                assert len(action) == len(id)
            for i, j in enumerate(id):
                # self.workers[j].send_action(action[i])
                self.workers[j].send_action(
                    {_agent: action[i] for _agent, action in action_n.items()}
                )
            result = []
            for j in id:
                obs, rew, done, info = self.workers[j].get_result()
                for a_id in self.agent_ids:
                    info[a_id]["env_id"] = j
                result.append((obs, rew, done, info))
        else:
            if action_n is not None:
                self._assert_id(id)
                for action in action_n.values():
                    assert len(action) == len(id)
                for i, j in enumerate(id):
                    self.workers[j].send_action(
                        {_agent: action[i] for _agent, action in action_n.items()}
                    )
                    self.waiting_conn.append(self.workers[j])
                    self.waiting_id.append(j)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[SubprocEnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                obs, rew, done, info = conn.get_result()
                for a_id in self.agent_ids:
                    info[a_id]["env_id"] = j
                result.append((obs, rew, done, info))
                self.ready_id.append(env_id)
        obs_list, rew_list, done_list, info_list = zip(*result)
        obs_dict, rew_dict, done_dict, info_dict = map(
            list_dict_to_dict_list, [obs_list, rew_list, done_list, info_list]
        )
        for a_id in obs_dict.keys():
            try:
                obs_dict[a_id] = np.stack(obs_dict[a_id])
            except ValueError:  # different len(obs)
                obs_dict[a_id] = np.array(obs_dict[a_id], dtype=object)
            rew_dict[a_id], done_dict[a_id], info_dict[a_id] = map(
                np.stack, [rew_dict[a_id], done_dict[a_id], info_dict[a_id]]
            )
            if self.obs_rms_n and self.update_obs_rms:
                self.obs_rms_n[a_id].update(obs_dict[a_id])
        # in case different agents share the same noramlizer, one should normlize obs
        # after all normalizers are updated.
        for a_id in obs_dict.keys():
            obs_dict[a_id] = self.normalize_obs(obs_dict[a_id], a_id)
        return obs_dict, rew_dict, done_dict, info_dict

    def seed(
        self, seed: Optional[Union[int, List[int]]] = None
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.
        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.
        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.
        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True

    def normalize_obs(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        """Normalize observations by statistics in obs_rms."""
        if self.obs_rms_n and self.norm_obs:
            clip_max = 10.0  # this magic number is from openai baselines
            # see baselines/common/vec_env/vec_normalize.py#L10
            obs = (obs - self.obs_rms_n[agent_id].mean) / np.sqrt(
                self.obs_rms_n[agent_id].var + self.__eps
            )
            obs = np.clip(obs, -clip_max, clip_max)
        return obs


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)


class Serializable(object):
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        Serializable.quick_init(self, locals())
        # self.action_space_n = self._wrapped_env.action_space_n
        # self.observation_space_n = self._wrapped_env.observation_space_n

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action_n):
        return self._wrapped_env.step(action_n)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self._wrapped_env, "terminate"):
            self._wrapped_env.terminate()

    def seed(self, seed):
        return self._wrapped_env.seed(seed)


def get_envs(
    scenarios,
    env_num=1,
    **kwargs,
):

    envs = SubprocVectorEnv(
        [lambda: ProxyEnv(SMARTSImitation(scenarios)) for _ in range(env_num)],
        **kwargs,
    )

    return envs


def get_env(scenarios):
    return ProxyEnv(SMARTSImitation(scenarios))


if __name__ == "__main__":
    kwargs = {}
    scenarios = [
        "/NAS2020/Workspaces/DRLGroup/syzhang/SMARTS_Imitation/smarts-imitation/ngsim"
    ]
    # env = ProxyEnv(SMARTSImitation(scenarios),**kwargs)
    env_num = 2
    step_num = 100
    env = get_env(scenarios)
    print("--------start creating envs--------")
    training_envs = get_envs(scenarios, env_num, **kwargs)
    training_envs.reset()
    for step in range(step_num):
        action = np.random.rand(2, 2)
        action = {"agent_0": action}
        next_ob, raw_reward, terminal, env_info = training_envs.step(action)
