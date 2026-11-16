# AquaNav-RL

AquaNav-RL is a lightweight and reproducible benchmark for autonomous
underwater vehicle (AUV) navigation in still water and uniform
ocean currents. The core simulator depends only on NumPy and Matplotlib. A
Gymnasium adapter and Stable-Baselines3 PPO training entry point are available
as optional dependencies.

The repository is intentionally small enough to read. It is designed for
students and researchers who want to test guidance, control, reinforcement
learning, robustness, and energy-aware planning without first installing a
large underwater simulator.

## Why this project

AUV navigation was the first research topic I explored after entering
university. At the time, I found very few reproducible open-source RL/DRL
projects for underwater navigation, and even fewer that supported
three-dimensional environments. Without a clear implementation to learn from
or build upon, I eventually had to pause the work.

This repository is the resource I wish I had then. It begins with a readable
planar 3-DOF baseline and is intended to grow toward spatial 5-DOF environments,
giving future students and researchers a clearer path to start, understand,
and extend AUV navigation research.

![Default benchmark](result/seed_7/trajectories.png)

## What is included

- nonlinear planar 3-DOF AUV dynamics;
- uniform and zero-current fields;
- configurable circular obstacles, observation noise, and mission geometry;
- a plain Python environment and an optional Gymnasium adapter;
- independently trained PPO policies for the two current scenarios;
- an optional PPO training script;
- trajectory, current-field, energy, and path-length visualization;
- deterministic evaluation and CSV export.

This public version is a benchmark and educational implementation. It does not
contain adaptive control-frequency selection, safety projection, learned ocean
current estimation, or communication-aware multi-AUV coordination.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[rl]'
python main.py --current none
```

The command trains the no-current PPO policy and creates:

```text
result/seed_7/none/
├── best_model.zip
├── final_model.zip
├── evaluation_history.csv
├── training_config.json
└── actual_timesteps.txt
```

Train both policies, then generate plots and GIFs:

```bash
bash run_ppo.sh
```

Train PPO:

```bash
python -m pip install -e '.[rl]'
python -m aquanav.train_ppo --steps 1000000 --current none
python -m aquanav.train_ppo --steps 1000000 --current uniform
```

Evaluate and visualize both trained policies:

```bash
python -m aquanav.evaluate_ppo --model-dir result --seed 7 --episodes 100
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
│   ├── evaluation.py      # deterministic rollout and metrics
│   ├── visualization.py   # plots and GIF generation
│   ├── train_ppo.py       # Stable-Baselines3 PPO training
│   └── evaluate_ppo.py    # trained-policy evaluation and visualization
├── tests/
├── main.py                # PPO training entry point
├── run_ppo.sh             # two-scenario training and evaluation
├── visualization.py
└── pyproject.toml
```

## Vehicle model

The state is split into inertial pose and body-frame velocity:

$$
\boldsymbol{\eta}=[x,y,\psi]^\top,\qquad
\boldsymbol{\nu}=[u,v,r]^\top.
$$

The kinematics and dynamics are

$$
\dot{\boldsymbol{\eta}}=
\mathbf J(\psi)\boldsymbol{\nu},
$$

$$
\mathbf M\dot{\boldsymbol{\nu}}+
\mathbf C(\boldsymbol{\nu}_r)\boldsymbol{\nu}_r+
\mathbf D(\boldsymbol{\nu}_r)\boldsymbol{\nu}_r
=\boldsymbol{\tau}.
$$

Here, $\boldsymbol{\nu}_r$ is the velocity relative to water. If the inertial
current velocity is $\mathbf v_c^n$, then

$$
\boldsymbol{\nu}_r=
\boldsymbol{\nu}-
\begin{bmatrix}
\mathbf R^\top(\psi)\mathbf v_c^n\\0
\end{bmatrix}.
$$

The action $\mathbf a\in[-1,1]^3$ is converted into surge force, sway force,
and yaw moment using the configured actuator limits. Nonlinear damping makes
the response speed-dependent. The state is integrated with fourth-order
Runge--Kutta at the default interval $\Delta t=0.1$ s.

## Current fields

The uniform field uses

$$
\mathbf v_c(\mathbf p,t)=V[\cos\theta,\sin\theta]^\top.
$$

Its velocity is constant in space and time, with default strength
$V=0.55$ m/s and direction $\theta=25^\circ$. The zero-current field returns
zero velocity.

## Navigation task

At step $t$, the observation is

$$
\mathbf o_t=
[\Delta x/W,\Delta y/W,\cos\psi,\sin\psi,
u/u_{\max},v/v_{\max},r/r_{\max},v_{c,x}^b,v_{c,y}^b,d/(2W)].
$$

The use of $\cos\psi$ and $\sin\psi$ avoids a discontinuity at $-\pi$ and
$\pi$. The current is represented in the body frame so that it has a direct
physical relationship with vehicle-relative velocity.

The reward is

$$
r_t=w_p(d_{t-1}-d_t)-w_t-w_e\|\mathbf a_t\|_2^2
+R_g\mathbb{1}_{\mathrm{goal}}
-R_c\mathbb{1}_{\mathrm{collision}\,\lor\,\mathrm{out\_of\_bounds}}.
$$

Progress provides a dense learning signal, while the energy term discourages
unnecessary thrust. A rollout ends after reaching the goal, colliding, leaving
the workspace, or exhausting the step budget.

## Baselines and evaluation

A reference LOS/PID controller points the vehicle toward the goal and reduces
forward speed when its heading error is large:

$$
u_d=u_{\mathrm{cruise}}\max(0,\cos e_\psi),
$$

$$
a_r=\operatorname{clip}\left(K_pe_\psi+K_i\int e_\psi\,dt+K_d\dot{e}_\psi,-1,1\right).
$$

These reference equations are not implemented in this version. Evaluation
uses the independently trained PPO policies. New algorithms should be
evaluated with repeated seeds and report:

- mission success and collision rate;
- final goal distance;
- path length and completion time;
- normalized control-energy proxy;
- mean and worst-case performance across current fields.

## Adding a controller

A controller used by the evaluation code needs a `predict` method:

```python
class MyController:
    def predict(self, observation, deterministic=True):
        ...
        return action, None  # NumPy array with shape (3,), values in [-1, 1]
```

It can then be passed as `model` to `aquanav.evaluation.run_episode(env, model, seed)`.

## Scope and limitations

The simulator is intended for algorithm prototyping, controlled comparisons,
and teaching. It is not a validated digital twin of a particular vehicle. It
does not model heave, roll, pitch, thruster allocation, pressure, detailed
sonar returns, actuator faults, or acoustic networking. Claims about real AUV
performance require parameter identification and hardware or high-fidelity
validation.

## Reproducibility checklist

- keep environment and evaluation seeds in result tables;
- evaluate the same scenario set for every controller;
- separate training randomization from the fixed test set;
- report failed missions instead of removing them;
- include configuration values and package versions;
- do not tune a controller on the final test scenarios.

## License

MIT.
