from time import perf_counter

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.dict import Dict

from envs.multi_agent_env import MultiAgentEnv
from envs.common import *
from envs.mubs_cov.maps import MAPS
from envs.mubs_cov.recorder import Recorder
from edgeric import EdgeRICScheduler


class MultiUbsCoverageEnv(MultiAgentEnv):
    """Multiple UAV Base stations (UBSs) providing downlink wireless coverage for ground terminals (GTs)"""

    h_ubs = 100.  # Elevation of UBSs (m)
    p_tx = 1e-3 * np.power(10, 10 / 10)  # Tx power (Watt)
    n0 = 1e-3 * np.power(10, -170 / 10)  # PSD of Rx noise (Watt/Hz)
    bw = 180e3  # Bandwidth of sub-channels (Hz)
    fc = 2.4e9  # Central carrier frequency (Hz)
    scene = 'dense-urban'  # Scene of channel model
    safe_dist = 10.  # Minimum distance between UBSs for collision avoidance (m)
    penalty = 5  # Amount of penalty on collision

    def __init__(self, map_id, fair_service=True, avoid_collision=True, record=True,
                 scheduler_mode="original", fast_scheduling=False, dt_sched=1.0,
                 record_substeps=False):
        super(MultiUbsCoverageEnv, self).__init__()

        if scheduler_mode not in ("original", "edgeric_pf"):
            raise ValueError("scheduler_mode must be 'original' or 'edgeric_pf'")
        self.scheduler_mode = scheduler_mode

        # Extract parameters specified by map.
        self.map = MAPS[map_id]
        map_params = self.map.get_params()
        for k, v in map_params.items():
            setattr(self, k, v)

        self.fast_scheduling = bool(fast_scheduling)
        self.dt_uav = self.dt  # Alias only: action amounts still use original dt.
        if isinstance(dt_sched, (bool, np.bool_)):
            raise ValueError("dt_sched must be a finite positive number of seconds")
        try:
            self.dt_sched = float(dt_sched)
        except (ValueError, TypeError):
            raise ValueError("dt_sched must be a finite positive number of seconds")
        if not np.isfinite(self.dt_sched) or self.dt_sched <= 0:
            raise ValueError("dt_sched must be a finite positive number of seconds")
        self.n_substeps = 1
        if self.fast_scheduling:
            ratio = self.dt_uav / self.dt_sched
            if (not np.isfinite(ratio) or ratio < 1 or
                    not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9)):
                raise ValueError("dt_uav must be an integer multiple of dt_sched")
            self.n_substeps = int(round(ratio))
        self.record_substeps = bool(record_substeps)

        self._fair_service = fair_service  # Whether fairness among GTs is considered
        self._avoid_collision = avoid_collision  # Whether penalties are imposed on collisions between UBSs

        # Create channel model and compute max possible link rate
        self.chan = AirToGroundChannel(self.scene, self.fc)  # Channel model
        g_max = self.chan.estimate_chan_gain(0, self.h_ubs)  # Maximum channel gain
        snr_max = self.p_tx * g_max / (self.n0 * self.bw)  # SNR of optimal link
        self.max_rate = self.bw * np.log2(1 + snr_max) * 1e-6  # Maximum link rate (Mbps)

        # Mbps regularizer: cold-start weights are bounded by 20 with this
        # channel model; do not use a near-zero machine epsilon for PF.
        self.edgeric_scheduler = (
            EdgeRICScheduler(self.bw, self.p_tx, self.n0, 0.05 * self.max_rate)
            if scheduler_mode == "edgeric_pf" else None
        )

        self.t = None  # Timer
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)  # Positions of UBSs (m)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)  # Positions of GTs (m)
        self.d_u2g = np.empty((self.n_ubs, self.n_gts), dtype=np.float32)  # Distance between UBSs and GTs (m)
        self.d_u2u = np.empty((self.n_ubs, self.n_ubs), dtype=np.float32)  # Distance between UBSs (m)
        self.adj = np.empty((self.n_ubs, self.n_ubs), dtype=bool)  # Adjacency matrix of agents
        self.prior_gts = np.empty(self.n_gts, dtype=int)  # Priority of GTs for receiving service
        self.sched = np.empty((self.n_ubs, self.n_gts), dtype=bool)  # Scheduling relations between UBSs and GTs
        self.mask_collision = np.empty(self.n_ubs, dtype=bool)

        self.rate_per_gt = np.empty(self.n_gts, dtype=np.float32)  # Instant data rate of each GT (Mbps)
        self.rate_per_ubs = np.empty(self.n_ubs, dtype=np.float32)  # Instant data rate offered by each UBS (Mbps)
        self.total_throughput = None  # Total throughput of the entire episode (Mb)
        self.n_colls = None  # Total number of collisions
        self.avg_rate_per_gt = np.empty(self.n_gts, dtype=np.float32)  # Average data rate of each GT (Mbps)
        self.fair_idx = None  # Jain's fairness index among GTs
        self.global_util = None  # Trade-off between fairness and throughout
        self.avg_global_util = None

        # Obtain possible actions.
        move_amounts = self.dt * np.array(self.vels).reshape(-1, 1)  # Possible amounts of movement in each timestep
        ang = 2 * np.pi * np.arange(self.n_dirs) / self.n_dirs  # Possible flying angles
        move_dirs = np.stack([np.cos(ang), np.sin(ang)]).T  # Moving directions of UBSs
        self.avail_moves = np.concatenate((np.zeros((1, 2)), np.kron(move_amounts, move_dirs)))  # Available moves

        self.n_agents = self.n_ubs  # Number of agents
        self.n_actions = self.avail_moves.shape[0]  # Number of actions

        # Define spaces of MARL
        self.observation_space = [Dict(spaces={
            'agent': Box(-np.inf, np.inf, shape=np.array([self.obs_own_feats_size])),
            'ubs': Box(-np.inf, np.inf, shape=np.array(self.obs_ubs_feats_size)),
            'gt': Box(-np.inf, np.inf, shape=np.array(self.obs_gt_feats_size))
        })] * self.n_agents  # Observation space of each agent
        self.state_space = Box(-np.inf, np.inf, shape=np.array([self.get_state_size()]))  # Space of global state
        self.action_space = [Discrete(self.n_actions)]  # Action space of each agent

        self.ep_ret = None  # Episode return

        # Tool for recording the episode.
        self.recorder = None
        if record:
            self.recorder = Recorder(self)

    def reset(self):
        self.t = 0  # Reset timer.
        self.ep_ret = 0  # Episode return
        self.avg_global_util = 0
        self.avg_rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)  # Average data rate of each GT (Mbps)
        self.total_throughput = 0  # Total throughput of episode (Mb)
        self.n_colls = 0  # Number of collisions

        self.scheduling_count = 0  # Actual radio decisions, independent of t.
        self.initialization_scheduling_count = 0
        self.service_time_s = 0.0
        self.scheduling_durations_s = []
        self.last_radio_trace = []
        self.delivered_mbit_per_gt = np.zeros(self.n_gts, dtype=np.float64)
        self.instant_rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)
        self.instant_rate_per_ubs = np.zeros(self.n_ubs, dtype=np.float32)

        if self.edgeric_scheduler is not None:
            self.edgeric_scheduler.reset()

        # Setup initial positions of UBSs and GTs.
        positions = self.map.set_positions()
        self.pos_ubs, self.pos_gts = positions['ubs'], positions['gt']
        self.prior_gts = np.random.permutation(self.n_gts)  # Randomly determine the initial GT priorities.
        self._transmit_data()  # UBSs provide wireless service for GTs.

        if self.recorder is not None:
            self.recorder.reload()

        return self.get_obs(), self.get_state()

    def step(self, actions):
        self.t += 1  # One timestep has elapsed.

        # Update UBS positions from actions.
        moves = self.avail_moves[np.array(actions, dtype=int)]  # Moves of all UBSs
        self.pos_ubs = np.clip(self.pos_ubs + moves, 0, self.range_pos)  # Latest positions of UBSs

        self._transmit_data()  # UBSs provide wireless service for GTs at new positions.

        # Get reward.
        reward = self._get_reward()
        self.ep_ret += reward.mean()  # Accumulate episode return.

        # Determine whether episode ends.
        done = self._get_terminate()
        # Summarize information of current env state.
        info = dict(EpRet=self.ep_ret, EpLen=self.t, AvgGlobalUtility=self.avg_global_util,
                    FairIdx=self.fair_idx, TotalThroughput=self.total_throughput, ProbCollision=self.n_colls / self.t)
        # Mark whether termination of episode is caused by reaching episode limit.
        info['BadMask'] = True if (self.t == self.episode_limit) else False

        # Record information.
        if self.recorder is not None:
            self.recorder.click(pos_ubs=self.pos_ubs.copy(), fair_idx=self.fair_idx, reward=reward.mean())

        return self.get_obs(), self.get_state(), reward, done, info

    def _transmit_data(self):
        """Dispatch legacy transmission or one independent radio service period."""
        self.last_radio_trace = []
        if self.fast_scheduling:
            if self.t == 0:
                # Initialization has no duration and makes no RB decision.
                self._update_geometry()
                self.sched = np.zeros((self.n_ubs, self.n_gts, self.n_rbs), dtype=bool)
                self.rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)
                self.rate_per_ubs = np.zeros(self.n_ubs, dtype=np.float32)
                self.last_prior_gts = self.prior_gts.copy()
                self.fair_idx = compute_jain_fairness_index(self.avg_rate_per_gt)
                self.global_util = 0.0
                return
            self._transmit_fast()
            return

        history = self.avg_rate_per_gt.copy() if self.record_substeps else None
        self._schedule_once()
        self.n_colls += self.mask_collision.sum() / 2
        # The following legacy expressions and reset service remain unchanged.
        self.avg_rate_per_gt = (self.avg_rate_per_gt * self.t + self.rate_per_gt) / (self.t + 1)  # Average data rate
        self.total_throughput += (self.rate_per_gt.sum() * self.dt / 1e3)  # Total throughput of network (Gb)
        self.fair_idx = compute_jain_fairness_index(self.avg_rate_per_gt)  # Long-term fairness index among GTs
        self.global_util = self.fair_idx * self.rate_per_gt.mean()  # Trade-off between fairness and throughout
        self.avg_global_util = (self.avg_global_util * self.t + self.global_util) / (self.t + 1)
        if self.scheduler_mode == "original":
            self.prior_gts = np.argsort(self.avg_rate_per_gt)  # Priorities of GTs for next timestep
        self.service_time_s += self.dt
        self.initialization_scheduling_count += int(self.t == 0)
        self.instant_rate_per_gt = self.rate_per_gt.copy()
        self.instant_rate_per_ubs = self.rate_per_ubs.copy()
        self._capture_radio_sample(self.dt, history)

    def _update_geometry(self):
        """Refresh distances, visibility and collision mask without counting time."""
        self.d_u2g = np.zeros((self.n_ubs, self.n_gts), dtype=np.float32)
        self.d_u2u = np.zeros((self.n_ubs, self.n_ubs), dtype=np.float32)
        for i in range(self.n_ubs):
            for m in range(self.n_gts):
                self.d_u2g[i, m] = np.linalg.norm(self.pos_gts[m] - self.pos_ubs[i])
            for j in range(self.n_ubs):
                self.d_u2u[i, j] = np.linalg.norm(self.pos_ubs[j] - self.pos_ubs[i])

        self.adj = (self.d_u2u <= self.r_comm)  # Get adjacency between agents.
        self.mask_collision = ((self.d_u2u + 99999 * np.eye(self.n_ubs)) < self.safe_dist).any(1)

    def _allocate_resources(self):
        """Stage-1 PF weights, RB allocation, interference and SINR, unchanged."""
        self.sched = np.zeros((self.n_ubs, self.n_gts, self.n_rbs), dtype=bool)
        p_itf = np.zeros((self.n_ubs, self.n_gts, self.n_rbs), dtype=np.float32)
        g = self.chan.estimate_chan_gain(self.d_u2g, self.h_ubs)  # Channel gain
        mask_itf = (self.d_u2g <= self.r_cov)  # Mask of interference range
        if self.edgeric_scheduler is not None:
            # Current positions/gains, previous completed transmission history.
            # On reset history is zero; the regularizer handles cold start.
            self.edgeric_scheduler.compute_weights(
                g, self.d_u2g, mask_itf, self.avg_rate_per_gt
            )
            self.prior_gts = self.edgeric_scheduler.get_priority()
        self.last_prior_gts = self.prior_gts.copy()
        for m in self.prior_gts:
            nearest_ubs = np.argsort(self.d_u2g[:, m])  # All UBSs from the nearest to the furthest
            for i in nearest_ubs:
                if (self.sched[i].sum() < self.n_rbs) and (self.d_u2g[i, m] <= self.r_cov):
                    # Identify idle RBs of UBS.
                    occupied_chan_idxes = np.where(self.sched[i].sum(0) > 0)
                    # Compute total interference at GT-m on each RB.
                    itf_per_chan = p_itf[:, m, :].sum(0)
                    # Select idle RB with the lowest interference level.
                    itf_per_chan[occupied_chan_idxes] = np.nan
                    opt_chan_idx = np.nanargmin(itf_per_chan)
                    self.sched[i, m, opt_chan_idx] = 1
                    # GTs receive interference on the selected RB if they lie in the coverage of UBS-i.
                    p_itf[i, :, opt_chan_idx] = self.p_tx * g[i] * mask_itf[i]
                    p_itf[i, m, opt_chan_idx] = 0
                    break
        
        # Compute data rates from channel gain and scheduling decisions.
        self.rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)
        for m in range(self.n_gts):
            if self.sched[:, m, :].sum() > 0:
                ubs_idx, chan_idx = np.where(self.sched[:, m, :])
                sinr = (self.p_tx * g[ubs_idx, m]) / (p_itf[:, m, chan_idx].sum() + self.bw * self.n0)
                self.rate_per_gt[m] = self.bw * np.log2(1 + sinr) * 1e-6  # Achievable rate of each GT (Mbps)
        self.rate_per_ubs = (self.sched.sum(-1) * self.rate_per_gt).sum(1)


    def _schedule_once(self):
        """Time CPU geometry + channel + priority + RB allocation + rate calculation.

        Excludes history/period aggregation and optional trace copies. There
        are no asynchronous GPU operations or wall-clock sleeps in this path.
        """
        started = perf_counter()
        self._update_geometry()
        self._allocate_resources()
        self.scheduling_durations_s.append(perf_counter() - started)
        self.scheduling_count += 1

    def _prepare_radio_substep(self, substep_index):
        """Version A holds the post-action UAV position for the entire period.

        Extension point for a future trajectory interpolation strategy.
        Such a strategy must also define collision aggregation; not enabled here.
        """
        pass

    def _transmit_fast(self):
        period_mbit_gt = np.zeros(self.n_gts, dtype=np.float64)
        period_mbit_ubs = np.zeros(self.n_ubs, dtype=np.float64)
        for substep in range(self.n_substeps):
            self._prepare_radio_substep(substep)
            history = self.avg_rate_per_gt.copy() if self.record_substeps else None
            self._schedule_once()
            # Collision convention remains collided-UAV count / 2, but only
            # once per UAV action (and never during fast reset).
            if substep == 0:
                self.n_colls += self.mask_collision.sum() / 2
            data_gt = self.rate_per_gt.astype(np.float64) * self.dt_sched
            data_ubs = self.rate_per_ubs.astype(np.float64) * self.dt_sched
            period_mbit_gt += data_gt
            period_mbit_ubs += data_ubs
            self.delivered_mbit_per_gt += data_gt
            self.service_time_s = self.scheduling_count * self.dt_sched
            self.avg_rate_per_gt = self.delivered_mbit_per_gt / self.service_time_s
            self.total_throughput = self.delivered_mbit_per_gt.sum() / 1e3
            if self.scheduler_mode == "original":
                self.prior_gts = np.argsort(self.avg_rate_per_gt)
            self._capture_radio_sample(self.dt_sched, history)

        # sched and PF diagnostics describe the last actual RB decision.
        # Observations and reward instead use time-average service of this
        # UAV period, including per-UAV averages for the idle-UAV reward mask.
        self.instant_rate_per_gt = self.rate_per_gt.copy()
        self.instant_rate_per_ubs = self.rate_per_ubs.copy()
        self.rate_per_gt = period_mbit_gt / self.dt_uav
        self.rate_per_ubs = period_mbit_ubs / self.dt_uav
        self.fair_idx = compute_jain_fairness_index(self.avg_rate_per_gt)
        self.global_util = self.fair_idx * self.rate_per_gt.mean()
        self.avg_global_util = (
            self.avg_global_util * (self.t - 1) + self.global_util
        ) / self.t

    def _capture_radio_sample(self, duration_s, history_before):
        """Optional last-period trace; never grows with episode length."""
        if not self.record_substeps:
            return
        sample = dict(duration_s=float(duration_s), service_time_s=self.service_time_s,
                      uav_step=self.t, rate_per_gt=self.rate_per_gt.copy(),
                      rate_per_ubs=self.rate_per_ubs.copy(), sched=self.sched.copy(),
                      history_before=history_before.copy(),
                      history_after=self.avg_rate_per_gt.copy(),
                      priority=self.last_prior_gts.copy(),
                      coverage=(self.d_u2g <= self.r_cov),
                      pos_ubs=self.pos_ubs.copy(),
                      compute_s=self.scheduling_durations_s[-1])
        if self.edgeric_scheduler is not None:
            sample["weights"] = self.edgeric_scheduler.weights.copy()
        self.last_radio_trace.append(sample)

    def get_scheduling_stats(self):
        """Measured compute latency, not a claim of a hard real-time deadline."""
        durations = np.asarray(self.scheduling_durations_s, dtype=np.float64)
        period = self.dt_sched if self.fast_scheduling else float(self.dt)
        return dict(dt_uav_s=float(self.dt), dt_sched_s=period,
                    scheduling_count=self.scheduling_count,
                    initialization_scheduling_count=self.initialization_scheduling_count,
                    service_time_s=self.service_time_s,
                    mean_compute_s=float(durations.mean()) if durations.size else 0.0,
                    p95_compute_s=float(np.percentile(durations, 95)) if durations.size else 0.0,
                    p99_compute_s=float(np.percentile(durations, 99)) if durations.size else 0.0,
                    max_compute_s=float(durations.max()) if durations.size else 0.0,
                    over_period_count=int((durations > period).sum()))

    def get_obs(self) -> list:
        return [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int) -> dict:
        """Returns local observation of specified agent as a dict."""
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        ubs_feats = np.zeros(self.obs_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        # Own features
        own_feats[0:2] = self.pos_ubs[agent_id] / self.range_pos

        # UBS features
        ubs_ids = [ubs_id for ubs_id in range(self.n_agents) if ubs_id != agent_id]  # IDs of other UAV-BSs
        for j, ubs_id in enumerate(ubs_ids):
            # if self.d_u2u[agent_id, ubs_id] <= self.r_sns:
            if self.d_u2u[agent_id, ubs_id] <= self.r_comm:
                ubs_feats[j, 0] = 1
                ubs_feats[j, 1:3] = (self.pos_ubs[ubs_id] - self.pos_ubs[agent_id]) / min(self.range_pos, self.r_comm)

        # GT features
        for m in range(self.n_gts):
            if self.d_u2g[agent_id, m] <= self.r_sns:
                gt_feats[m, 0] = 1
                gt_feats[m, 1:3] = (self.pos_gts[m] - self.pos_ubs[agent_id]) / min(self.range_pos, self.r_sns)
                gt_feats[m, 3] = self.rate_per_gt[m] / self.max_rate
                if self._fair_service:
                    # gt_feats[m, 4] = self.avg_rate_per_gt[m] / max(1e-5, self.avg_rate_per_gt.sum())
                    gt_feats[m, 4] = self.avg_rate_per_gt[m] / self.max_rate * self.n_gts / (self.n_ubs * self.n_rbs)

        return dict(agent=own_feats, ubs=ubs_feats, gt=gt_feats)

    def get_obs_size(self) -> dict:
        return dict(agent=self.obs_own_feats_size, ubs=self.obs_ubs_feats_size, gt=self.obs_gt_feats_size)

    @property
    def obs_own_feats_size(self) -> int:
        """
        Features of agent itself include:
        - Normalized position (x, y)
        """
        nf_own = 2
        return nf_own

    @property
    def obs_ubs_feats_size(self) -> tuple:
        """
        Observed features of each UBS include
        - Visibility flag
        - Normalized distance (x, y) when visible
        """
        nf_ubs = 1 + 2
        return self.n_ubs - 1, nf_ubs

    @property
    def obs_gt_feats_size(self) -> tuple:
        """
        Observed features of each GT include
        - Visibility flag
        - Normalized distance (x, y) when visible
        - Normalized instance data rate when visible
        - Normalized average data rate when visible and fairness is considered
        """
        nf_gt = 1 + 2 + 1
        if self._fair_service:
            nf_gt += 1
        return self.n_gts, nf_gt

    def get_state(self) -> np.ndarray:
        """
        Returns features of all UBSs and GTs as global env state.
        Note that state is only used for centralized training and should be inaccessible during inference.
        """
        ubs_feats = np.zeros(self.state_ubs_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.state_gt_feats_size, dtype=np.float32)

        # Features of UBSs
        ubs_feats[:, 0:2] = self.pos_ubs / self.range_pos

        # Features of GTs
        gt_feats[:, 0:2] = self.pos_gts / self.range_pos
        gt_feats[:, 2] = self.rate_per_gt / self.max_rate  # Normalized instant rates
        if self._fair_service:
            gt_feats[:, 3] = self.avg_rate_per_gt / self.max_rate * self.n_gts / (self.n_ubs * self.n_rbs)

        return np.concatenate((ubs_feats.flatten(), gt_feats.flatten()))

    def get_state_size(self) -> int:
        return np.prod(self.state_ubs_feats_size) + np.prod(self.state_gt_feats_size)

    @property
    def state_ubs_feats_size(self) -> tuple:
        """
        State of each UBS includes
        - Normalized position (x, y)
        """
        nf_ubs = 2
        return self.n_ubs, nf_ubs

    @property
    def state_gt_feats_size(self) -> tuple:
        """
        State of each GT includes
        - Normalized position (x, y)
        - Normalized instant data rate
        - Normalized average data rate when fairness is considered
        """
        nf_gt = 2 + 1
        if self._fair_service:
            nf_gt += 1
        return self.n_gts, nf_gt

    def _get_reward(self):
        """Computes local reward of each individual agent."""

        if self._fair_service:
            local_rewards = self.global_util * np.ones(self.n_agents, dtype=np.float32)
        else:
            local_rewards = self.rate_per_gt.mean() * np.ones(self.n_agents, dtype=np.float32)

        # Scale reward to appropriate magnitude.
        local_rewards = self.reward_scale_rate * local_rewards / self.max_rate
        idle_ubs_mask = (self.rate_per_ubs == 0)  # Indices of UBSs serving no GT
        local_rewards = local_rewards * (1 - idle_ubs_mask)  # Idle UBSs do not receive rewards.

        # Each agent is penalized for its collision with other UBSs.
        if self._avoid_collision:
            local_rewards = (1 - self.mask_collision) * local_rewards - self.mask_collision * self.penalty

        return local_rewards

    def _get_terminate(self) -> bool:
        """Determines the end of an episode."""
        return True if (self.t == self.episode_limit) else False

    def replay(self, **kwargs):
        """Replays the trajectories of UBSs throughout the episode."""
        if self.recorder is not None:
            self.recorder.replay(**kwargs)


if __name__ == '__main__':
    env = MultiUbsCoverageEnv(map_id='test')
    # print(env.__dict__)

    env.reset()
    env.r_cov = np.inf
    env.pos_gts = np.array([[200, 200]])
    for t in range(40):
        env.pos_ubs = np.array([[10 * t, 200]])
        env.step([0])
        print(env.d_u2g)
        print(env.rate_per_gt)