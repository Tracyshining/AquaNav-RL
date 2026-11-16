# AquaNav-RL

AquaNav-RL is a lightweight, reproducible framework for autonomous underwater vehicle (AUV) navigation under spatially and temporally varying ocean currents. The core simulator is built with NumPy and Matplotlib, with optional Gymnasium integration and PPO training through Stable-Baselines3.

The project provides an accessible environment for students and researchers to explore underwater navigation, guidance, control, and reinforcement learning. Its modular structure supports the development and comparison of different navigation algorithms.

## Why I Built This Project

AUV navigation was the first research topic I explored after entering
university. At the time, I found very few reproducible open-source RL/DRL
projects for underwater navigation, and even fewer that supported
three-dimensional environments. Without a clear implementation to learn from
or build upon, I eventually had to pause the work.

This repository is the resource I wish I had then. It begins with a readable
planar 3-DOF baseline and is intended to grow toward spatial 5-DOF environments,
giving future students and researchers a clearer path to start, understand,
and extend AUV navigation research.


## What is included

- Nonlinear planar 3-DOF AUV dynamics
- Uniform, vortex, oscillating, and zero-current fields
- Configurable circular obstacles, observation noise, and mission geometry
- Python simulation environment with Gymnasium support
- LOS/PID and random-controller baselines
- PPO training and evaluation scripts
- Trajectory, current-field, energy, and path-length visualization
- Reproducible evaluations with CSV exports
- Mathematical formulation in `docs/methodology.tex`


## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python main.py
```

The command evaluates the LOS/PID baseline in four current fields and creates:

```text
artifacts/
├── benchmark.png
└── summary.csv
```

Generate a GIF for the vortex-current episode:

```bash
python main.py --currents vortex --animate
```

Train PPO:

```bash
python -m pip install -e '.[rl]'
python -m aquanav.train_ppo --steps 300000 --current vortex
```

Evaluate and visualize a trained policy:

```bash
python -m aquanav.evaluate_ppo --model models/ppo_auv --current vortex
```

## PPO algorithm and reference

Navigation policies are trained with **Proximal Policy Optimization (PPO-Clip)**,
using the standard Stable-Baselines3 implementation. For the original algorithm,
see **Schulman et al. (2017), *Proximal Policy Optimization Algorithms***:
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
([PDF](https://arxiv.org/pdf/1707.06347)).

The relevant equations from the original paper are identified directly above
the PPO configuration and training calls in
[`aquanav/train_ppo.py`](aquanav/train_ppo.py). For the precise mapping from
paper equations to the code and the underlying SB3 implementation, see
[`docs/PPO_Theory.md`](docs/PPO_Theory.md). The PPO loss itself is calculated
inside Stable-Baselines3, rather than reimplemented in this repository.

**Suggested citation**

```bibtex
@article{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```


## Repository layout

```text
AquaNav-RL/
├── aquanav/
│   ├── config.py          # physical and task parameters
│   ├── currents.py        # current-field models
│   ├── dynamics.py        # 3-DOF equations and RK4 integration
│   ├── environment.py     # simulator-independent task API
│   ├── gym_env.py         # optional Gymnasium adapter
│   ├── controllers.py     # LOS/PID and random baselines
│   ├── evaluation.py      # deterministic rollout and metrics
│   ├── visualization.py   # plots and GIF generation
│   ├── main.py            # benchmark entry point
│   ├── train_ppo.py       # Stable-Baselines3 PPO training
│   └── evaluate_ppo.py    # trained-policy evaluation and visualization
├── docs/methodology.tex
├── tests/
├── main.py
├── visualization.py
└── pyproject.toml
```

## Vehicle model

The AUV state consists of the inertial-frame pose and body-frame velocity, defined as

$$
\boldsymbol{\eta}=[x,y,\psi]^\top,\qquad
\boldsymbol{\nu}=[u,v,r]^\top.
$$

where $x$ and $y$ represent the vehicle position, $\psi$ is the yaw angle, and $u$, $v$, and $r$ denote the surge velocity, sway velocity, and yaw rate, respectively.

The vehicle kinematics are formulated as

$$
\dot{\boldsymbol{\eta}}=
\mathbf J(\psi)\boldsymbol{\nu},
$$

where $\mathbf{J}(\psi)$ is the coordinate transformation matrix from the body-fixed frame to the inertial frame.

The nonlinear vehicle dynamics are expressed as
$$
\mathbf M\dot{\boldsymbol{\nu}}+
\mathbf C(\boldsymbol{\nu}_r)\boldsymbol{\nu}_r+
\mathbf D(\boldsymbol{\nu}_r)\boldsymbol{\nu}_r
=\boldsymbol{\tau}.
$$

where $\mathbf{M}$ is the inertia matrix, $\mathbf{C}$ represents the Coriolis and centripetal matrix, $\mathbf{D}$ denotes the hydrodynamic damping matrix, and $\boldsymbol{\tau}$ is the generalized control force vector.

The water-relative velocity is defined as

$$
\boldsymbol{\nu}_r =
\boldsymbol{\nu} -
\left[
\begin{array}{c}
\mathbf{R}^{\mathrm{T}}(\psi)\mathbf{v}_c^n \\[8pt]
0
\end{array}
\right]
$$


where $\mathbf{v}_c^n$ represents the ocean current velocity in the inertial frame, and $\mathbf{R}(\psi)$ is the planar rotation matrix.

The normalized control action $\mathbf{a}\in[-1,1]^3$ is mapped to surge force, sway force, and yaw moment according to the configured actuator limits. The vehicle state is integrated using the fourth-order Runge–Kutta method with a default time step of $\Delta t=0.1$ s.


## Current fields

The vortex current field is formulated as


$$
\mathbf{v}_c(\mathbf{p}) =
\begin{cases}
V \cdot \dfrac{2R}{\rho^2+R^2}
\cdot [-d_y,d_x]^\top, & \rho>0,\\[8pt]
\mathbf{0}, & \rho=0.
\end{cases}
$$


where $V$ is the nominal current strength, $R$ represents the vortex core radius, and $\rho$ is the radial distance from the vortex center. The relative position and radial distance are defined as

$$
\mathbf{d} = \mathbf{p}-\mathbf{c}
$$

$$
\rho = \|\mathbf{d}\|.
$$

where $\mathbf{p}$ denotes the vehicle position and $\mathbf{c}$ is the vortex center.

The current velocity approaches zero at the vortex center, reaches its nominal strength near the core radius, and decreases with increasing distance. The oscillating current field incorporates time-dependent direction and position-dependent phase to represent non-stationary ocean disturbances.

## Navigation task

At each time step $t$, the observation vector is defined as

$$
\mathbf{o}_t =
\left[
\frac{\Delta x}{W},\;
\frac{\Delta y}{W},\;
\cos\psi,\;
\sin\psi,\;
\frac{u}{u_{\max}},\;
\frac{v}{v_{\max}},\;
\frac{r}{r_{\max}},\;
v_{c,x}^b,\;
v_{c,y}^b,\;
\frac{d}{2W}
\right].
$$

where $\Delta x$ and $\Delta y$ represent the relative position of the goal, $W$ is the workspace scale, and $u_{\max}$, $v_{\max}$, and $r_{\max}$ are the velocity normalization limits. The terms $v_{c,x}^b$ and $v_{c,y}^b$ denote the body-frame current velocity components, while $d$ represents the distance to the goal.

The heading angle is represented by $\cos\psi$ and $\sin\psi$ to maintain continuity across the angular boundary. The body-frame current components provide information about the disturbance relative to the vehicle orientation.

The reward function is formulated as


$$
r_t =
w_p \cdot (d_{t-1}-d_t)
-w_t
-w_e \cdot \|\mathbf{a}_t\|^2
+R_g \cdot \mathbb{I}_{\mathrm{goal}}
-R_c \cdot \mathbb{I}_{\mathrm{failure}}.
$$



where $d_{t-1}$ and $d_t$ denote the distances to the goal at consecutive time steps, $w_p$ is the progress reward coefficient, $w_t$ is the time penalty, and $w_e$ is the control-energy penalty coefficient. The parameters $R_g$ and $R_c$ represent the goal reward and failure penalty, respectively. The indicator functions $\mathbb{I}_{\mathrm{goal}}$ and $\mathbb{I}_{\mathrm{failure}}$ equal one when the vehicle reaches the goal or fails due to collision or workspace violation, respectively, and zero otherwise.


The episode terminates when the vehicle reaches the goal, collides with an obstacle, leaves the workspace, or exceeds the maximum number of simulation steps.

## Control and Evaluation

The LOS/PID controller determines the desired forward velocity based on the heading error, expressed as

$$
u_d = u_{\mathrm{cruise}} \cdot \max(0,\cos e_\psi).
$$

where $u_{\mathrm{cruise}}$ is the nominal cruising speed and $e_\psi$ represents the heading error relative to the goal direction.

The normalized yaw control action is formulated as

$$
a_r =
\mathrm{clip}
\left(
K_p \cdot e_\psi
+ K_i \cdot \int e_\psi\,dt
+ K_d \cdot \dot{e}_\psi,
-1,\,1
\right).
$$

where $K_p$, $K_i$, and $K_d$ denote the proportional, integral, and derivative gains, respectively. The function $\mathrm{clip}(\cdot,-1,1)$ limits the yaw control action to the interval $[-1,1]$.




## Custom Controllers  *( if you need )*

A controller only needs two methods:

```python
class MyController:
    def reset(self):
        ...

    def act(self, state, goal, dt):
        return action  # NumPy array with shape (3,), values in [-1, 1]
```

It can then be passed directly to `aquanav.evaluation.run_episode`.


## License

MIT.
