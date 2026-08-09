# DQN on CartPole: the maths, step by step

This is a companion to [`dqn.ipynb`](./dqn.ipynb). It explains what the code is
trying to calculate, how to read each equation, and how the equations become
PyTorch operations.

The central idea is:

> The policy network estimates how much future reward each action will lead to.
> We train it by comparing its estimate with the immediate reward plus a
> slow-moving estimate of the reward available from the next state.

The implementation follows the
[PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
and uses Gymnasium's
[CartPole-v1 environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

## 1. The pieces of the problem

At each timestep, the agent observes the environment, chooses an action, and
receives a result.

We use these symbols:

| Symbol | Read as | Meaning in the notebook |
|---|---|---|
| $t$ | “time step t” | One interaction with CartPole |
| $s_t$ | “state at time t” | The four-number observation |
| $a_t$ | “action at time t” | Push left (`0`) or right (`1`) |
| $r_t$ | “reward at time t” | `+1` for surviving that step |
| $s_{t+1}$ | “state at time t plus one” | The observation after the action |
| $\gamma$ | “gamma” | Discount factor, `GAMMA = 0.99` |
| $Q(s,a)$ | “Q of state s and action a” | Estimated future reward |

### The state

The CartPole state is a vector of four numbers:

$$
s_t =
\begin{bmatrix}
\text{cart position} \\
\text{cart velocity} \\
\text{pole angle} \\
\text{pole angular velocity}
\end{bmatrix}
$$

Read this as: “The state at time $t$ is a vector containing the cart position,
cart velocity, pole angle, and pole angular velocity.”

In the notebook, one state has shape `[1, 4]`:

```python
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
```

The original Gymnasium observation has shape `[4]`. `unsqueeze(0)` adds the
batch dimension, producing `[1, 4]`.

### The action

There are two possible actions:

$$
a_t \in \{0, 1\}
$$

Read this as: “The action at time $t$ is one of the values zero or one.”

- $a_t=0$: push the cart left.
- $a_t=1$: push the cart right.

### A transition

One piece of experience is:

$$
(s_t, a_t, r_t, s_{t+1})
$$

Read this as: “In state $s_t$, the agent took action $a_t$, received reward
$r_t$, and arrived at state $s_{t+1}$.”

The notebook stores exactly this information:

```python
Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward"),
)
```

The field order is slightly different, but the information is the same.

## 2. Reward is not the same as return

The **reward** is the immediate feedback from one step. In CartPole it is `+1`
for each step, including the terminating step.

The **return** is the reward from now into the future:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots
$$

Read this as:

> “The return from time $t$ equals the reward now, plus gamma times the next
> reward, plus gamma squared times the reward after that, and so on.”

The exponent tells us how far into the future a reward is. A reward one step
away is multiplied by $\gamma$; a reward two steps away is multiplied by
$\gamma^2$.

With $\gamma=0.99$, three rewards of `+1` are worth:

$$
G_t = 1 + 0.99(1) + 0.99^2(1) = 2.9701
$$

Read this as: “One now, plus ninety-nine percent of the next one, plus
ninety-nine percent squared of the third one.”

Because $\gamma$ is close to `1`, the agent cares strongly about surviving far
into the future. Discounting still makes nearby rewards slightly more valuable
and keeps an infinite sum well behaved.

## 3. What the Q-network predicts

The Q-value is the expected return after taking an action in a state:

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi\left[G_t \mid s_t=s,\ a_t=a\right]
$$

Read this as:

> “Q under policy pi, for state $s$ and action $a$, is the expected future
> return, given that the current state is $s$ and the current action is $a$.”

The vertical bar $\mid$ means “given that.” The $\mathbb{E}$ means an expected
or average value over possible future experiences. The superscript $\pi$ refers
to the policy—the rule the agent follows when choosing future actions.

The network takes four state values and produces two Q-values:

$$
Q_\theta(s_t)
=
\begin{bmatrix}
Q_\theta(s_t,\text{left}) & Q_\theta(s_t,\text{right})
\end{bmatrix}
$$

Read this as: “The Q-network with parameters theta takes state $s_t$ and
returns one value for left and one value for right.”

$\theta$ (“theta”) means all the trainable weights and biases in `policy_net`.
For example, the network might return:

```text
left:  18.4
right: 27.1
```

These values are **not probabilities**. They are estimates of discounted future
reward, so they do not have to be between zero and one or add up to one.

The notebook's network performs:

```text
[4 state values] -> [128 hidden values] -> [128 hidden values] -> [2 Q-values]
```

## 4. Choosing an action: exploration and exploitation

If the agent always trusted an untrained network, it might keep repeating a bad
action and never discover anything better. DQN therefore uses an
**epsilon-greedy policy**.

$$
a_t =
\begin{cases}
\text{random action}, & \text{with probability } \varepsilon_t \\
\displaystyle\arg\max_a Q_\theta(s_t,a),
& \text{with probability } 1-\varepsilon_t
\end{cases}
$$

Read this as:

> “At time $t$, choose a random action with probability epsilon. Otherwise,
> choose the action whose Q-value is largest.”

The expression $\arg\max_a$ is read “arg max over actions.” `max` asks for the
largest value, while `argmax` asks which action produced that value.

This line implements the greedy branch:

```python
policy_net(state).max(1).indices.view(1, 1)
```

### Epsilon decay

The probability of a random action decays over interaction steps:

$$
\varepsilon_t
=
\varepsilon_{\text{end}}
+
(\varepsilon_{\text{start}}-\varepsilon_{\text{end}})
e^{-t/d}
$$

Read this as:

> “Epsilon at step $t$ equals the final epsilon plus the difference between
> starting and final epsilon, multiplied by $e$ raised to negative $t$ divided
> by the decay constant.”

Here, $e$ is the exponential constant, approximately `2.718`, and $d$ is
`EPS_DECAY`. The negative exponent makes the extra exploration shrink as $t$
grows.

With the notebook's settings:

| `steps_done` | Approximate $\varepsilon_t$ | Meaning |
|---:|---:|---|
| 0 | 0.900 | 90% random actions |
| 2,500 | 0.337 | 33.7% random actions |
| 5,000 | 0.130 | 13.0% random actions |
| 10,000 | 0.026 | 2.6% random actions |

This is why `steps_done += 1` matters. Without it, $t$ stays at zero and the
agent chooses random actions 90% of the time forever.

## 5. The Bellman equation: the key idea

Suppose we knew the perfect Q-function, $Q^*$. Its value would satisfy:

$$
Q^*(s_t,a_t)
=
r_t
+
\gamma \max_{a'} Q^*(s_{t+1},a')
$$

Read this as:

> “The best possible value of taking action $a_t$ in state $s_t$ equals the
> reward received now, plus gamma times the value of the best action available
> in the next state.”

The prime in $a'$ is read “a prime.” It just distinguishes a possible **next**
action from the action $a_t$ that was already taken.

This equation is recursive: the value now is defined using the value one step
later. If the next-state Q-value is itself accurate, it already summarizes the
rewards after that next state. We therefore do not need to wait until the end
of every episode before making an update.

This process—improving one estimate using another estimate—is called
**bootstrapping**.

## 6. The prediction and the training target

The perfect $Q^*$ is unknown. DQN uses two approximations:

- $Q_\theta$: `policy_net`, with parameters $\theta$.
- $Q_{\bar\theta}$: `target_net`, with older parameters $\bar\theta$.

The policy network's prediction for one stored transition is:

$$
\hat q_t = Q_\theta(s_t,a_t)
$$

Read this as: “Q-hat at time $t$ is the policy network's estimated value of the
action that was actually taken.” A hat commonly means “an estimate.”

The target is:

$$
y_t =
\begin{cases}
r_t + \gamma \displaystyle\max_{a'}Q_{\bar\theta}(s_{t+1},a'),
& \text{if }s_{t+1}\text{ is non-terminal} \\
r_t, & \text{if }s_{t+1}\text{ is terminal}
\end{cases}
$$

Read this as:

> “The target $y_t$ is the reward plus the discounted best next-state value. If
> the episode truly terminated, the target is only the reward because no future
> state exists.”

The target is not a perfect label supplied by a human. It is a more useful
estimate built from an observed reward and the slower target network.

### A numerical example

Imagine a replay-memory item where the agent chose right:

```text
policy_net(state)       = [7.2, 8.0]
action taken            = 1 (right)
target_net(next_state)  = [9.0, 12.0]
reward                  = 1
gamma                   = 0.99
```

The prediction is the Q-value for the action actually taken:

$$
\hat q_t = Q_\theta(s_t,1) = 8.0
$$

The best next-state value is:

$$
\max(9.0,12.0)=12.0
$$

The target is therefore:

$$
y_t = 1 + 0.99(12.0) = 12.88
$$

The policy predicted `8.0`, but the target is `12.88`. Training adjusts the
policy network so its prediction for this kind of state-action pair moves
toward `12.88`.

If the action had caused termination, the target would instead be:

$$
y_t = r_t = 1
$$

That pushes an overestimated action value downward. This is how failure becomes
informative even though CartPole still returns `+1` on the terminating step:
the failed action has no estimated future rewards after that `+1`.

## 7. How `optimize_model()` implements the maths

### Step 1: wait for enough experience

```python
if len(memory) < BATCH_SIZE:
    return
```

No optimization happens until replay memory contains at least 128 transitions.

### Step 2: sample a random batch

```python
transitions = memory.sample(BATCH_SIZE)
batch = Transition(*zip(*transitions))
```

`transitions` starts as a list of complete rows:

```text
[(s1, a1, s2, r1), (s2, a2, s3, r2), ...]
```

The `zip` operation transposes it into columns:

```text
states      = [s1, s2, ...]
actions     = [a1, a2, ...]
next_states = [s2, s3, ...]
rewards     = [r1, r2, ...]
```

### Step 3: separate terminal and non-terminal states

```python
non_final_mask = torch.tensor(
    tuple(s is not None for s in batch.next_state),
    device=device,
    dtype=torch.bool,
)
```

For example:

```text
next states: [state, None, state, None]
mask:        [ True, False, True, False]
```

`None` means there is no next state because the pole fell or the cart left its
allowed range.

### Step 4: construct the batch tensors

```python
state_batch = torch.cat(batch.state)
action_batch = torch.cat(batch.action)
reward_batch = torch.cat(batch.reward)
```

The important shapes are:

| Tensor | Shape | Meaning |
|---|---:|---|
| `state_batch` | `[128, 4]` | 128 states, four values each |
| `policy_net(state_batch)` | `[128, 2]` | left and right Q-values for each state |
| `action_batch` | `[128, 1]` | action actually taken in each transition |
| `state_action_values` | `[128, 1]` | Q-value of each action actually taken |
| `reward_batch` | `[128]` | observed reward for each transition |
| `next_state_values` | `[128]` | best target-network value of each next state |

### Step 5: select the predictions for the actions taken

```python
state_action_values = policy_net(state_batch).gather(1, action_batch)
```

Suppose the network output and stored actions were:

```text
Q-values: [[4, 7],
           [3, 2]]

actions:  [[1],
           [0]]
```

`gather` selects column `1` from the first row and column `0` from the second:

```text
selected Q-values: [[7],
                    [3]]
```

We directly supervise only the action that was actually taken because the
transition tells us what happened after that action. We did not observe what
would have happened if the other action had been chosen.

### Step 6: calculate next-state values

```python
next_state_values = torch.zeros(BATCH_SIZE, device=device)
with torch.no_grad():
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1).values
    )
```

The tensor begins with zeros, so terminal transitions automatically have zero
future value. For non-terminal transitions, the target network predicts both
actions and `.max(1).values` selects the larger Q-value.

`torch.no_grad()` is essential: the target is treated as a fixed number for
this update. Gradients must update `policy_net`, not flow backward through
`target_net`.

### Step 7: build the Bellman targets

```python
expected_state_action_values = (
    next_state_values * GAMMA
) + reward_batch
```

This is the Bellman target:

$$
y_t = r_t + \gamma \max_{a'}Q_{\bar\theta}(s_{t+1},a')
$$

For terminal transitions, `next_state_values` is zero, giving $y_t=r_t$.

## 8. Measuring the error

For batch item $i$, define the temporal-difference error:

$$
\delta_i = \hat q_i-y_i
$$

Read this as: “Delta for item $i$ equals the policy prediction minus the target.”

Some books define delta with the opposite sign, $y_i-\hat q_i$. Either convention
describes the gap; this loss is symmetric, so the sign does not change the loss
value.

The notebook uses Huber loss, implemented by `SmoothL1Loss`:

$$
\ell(\delta_i)=
\begin{cases}
\frac{1}{2}\delta_i^2, & |\delta_i|\leq 1 \\
|\delta_i|-\frac{1}{2}, & |\delta_i|>1
\end{cases}
$$

Read this as:

> “If the absolute error is at most one, use half the squared error. If the
> absolute error is greater than one, use the absolute error minus one half.”

The notation $|\delta_i|$ means the absolute value of the error. Huber loss is
quadratic for small errors, giving smooth corrections, and linear for large
errors, making unusually large and noisy Q-errors less explosive.

The batch loss is the average over all $B=128$ samples:

$$
L(\theta)=\frac{1}{B}\sum_{i=1}^{B}\ell(\delta_i)
$$

Read this as: “The loss for parameters theta is one over the batch size times
the sum of each sample's Huber loss.”

The corresponding code is:

```python
criterion = nn.SmoothL1Loss()
loss = criterion(
    state_action_values,
    expected_state_action_values.unsqueeze(1),
)
```

`unsqueeze(1)` changes the targets from shape `[128]` to `[128, 1]`, matching
the predictions.

## 9. How the policy network learns

Conceptually, gradient descent performs:

$$
\theta \leftarrow \theta-\alpha\nabla_\theta L(\theta)
$$

Read this as:

> “Replace theta with its current value minus the learning rate alpha times the
> gradient of the loss with respect to theta.”

The arrow $\leftarrow$ means “is updated to.” The symbol $\nabla$ (“nabla”) means
the gradient: the direction in parameter space that increases the loss most.
Subtracting it moves toward lower loss.

Your notebook uses AdamW rather than plain gradient descent, so it adaptively
scales the update for each parameter. The high-level idea is still “change the
weights in the direction that reduces the Q-value error.”

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
optimizer.step()
```

These lines mean:

1. `zero_grad()`: remove gradients left from the previous update.
2. `backward()`: use the chain rule to calculate how every policy parameter
   affected the loss.
3. `clip_grad_value_()`: limit extreme gradients for stability.
4. `step()`: update the policy parameters.

The environment itself is not differentiated through. Gymnasium produces
experience; PyTorch differentiates only through the policy network's
calculation of $Q_\theta(s_t,a_t)$.

## 10. Why the target network exists

If `policy_net` created both the prediction and the target, each optimizer step
would change both sides of the learning problem. The network would be chasing a
target that moves immediately whenever it moves.

`target_net` is a delayed copy. It is not attached to an optimizer and does not
learn through `loss.backward()`. Instead, every environment step performs a
soft update:

$$
\bar\theta
\leftarrow
\tau\theta+(1-\tau)\bar\theta
$$

Read this as:

> “Update the target parameters to tau times the policy parameters plus one
> minus tau times the old target parameters.”

With $\tau=0.005$:

$$
\bar\theta
\leftarrow
0.005\theta+0.995\bar\theta
$$

So each target parameter moves only 0.5% of the way toward the corresponding
policy parameter on each step.

For example, if one policy parameter is `2.0` and its target-network copy is
`1.0`:

$$
0.005(2.0)+0.995(1.0)=1.005
$$

The target copy moves from `1.0` to only `1.005`, providing a slowly changing
training target.

The notebook implements this for every parameter tensor:

```python
for key in policy_net_state_dict:
    target_net_state_dict[key] = (
        policy_net_state_dict[key] * TAU
        + target_net_state_dict[key] * (1 - TAU)
    )
```

A useful mental model is:

```text
policy_net  = student being trained
target_net  = delayed copy used to prepare stable exercises
```

The target network is not an independently intelligent teacher. Everything it
knows originally came from the policy network; it is useful because it changes
more slowly.

## 11. Why replay memory exists

The agent generates a stream of strongly related experiences:

```text
state at step 20
state at step 21
state at step 22
```

Training only on consecutive states can make updates biased toward the most
recent part of one episode. Replay memory stores up to 10,000 transitions and
samples 128 of them randomly.

This provides two benefits:

1. Old experience can be reused instead of being discarded after one update.
2. A random batch mixes different moments and episodes, reducing correlation
   between neighboring training samples.

Unlike a normal supervised-learning dataset, this dataset is created by the
model's own behaviour. As the policy changes, the kinds of states and actions
stored in memory change too.

## 12. The complete training loop

Each episode begins with:

```python
state, info = env.reset()
```

Then each timestep does the following:

1. `select_action(state)` chooses randomly or uses `policy_net`.
2. `env.step(action.item())` returns the next observation and reward.
3. The transition is added to replay memory.
4. `optimize_model()` trains `policy_net` on a random batch.
5. `target_net` moves slightly toward `policy_net`.
6. The next state becomes the current state.
7. If the episode ended, its duration is recorded and a new episode begins.

In compact mathematical form:

$$
s_t
\xrightarrow{\text{choose }a_t}
(r_t,s_{t+1})
\xrightarrow{\text{store}}
\text{replay memory}
\xrightarrow{\text{sample}}
\text{policy update}
$$

Read this from left to right:

> “Starting in state $s_t$, choose action $a_t$, observe reward $r_t$ and the
> next state, store the transition, sample replay memory, and update the policy.”

### `terminated` versus `truncated`

The code ends an episode for either condition:

```python
done = terminated or truncated
```

- `terminated` means the environment reached a true terminal state, such as the
  pole falling. The code sets `next_state = None`, so future value is zero.
- `truncated` means an outside time limit ended the episode. CartPole-v1 is
  truncated at 500 steps. The code retains the next state when building the
  target because the underlying situation was not itself terminal.

## 13. What the duration plot measures

CartPole gives one reward for every step, so the undiscounted episode reward is
the episode duration:

$$
\text{episode reward}
=
\sum_{t=1}^{T}1
=
T
$$

Read this as: “The episode reward is the sum of one over all $T$ steps, which
equals $T$.”

That is why plotting `episode_durations` is a direct way to see behaviour
improve. CartPole-v1 has a 500-step time limit, so repeatedly reaching 500 means
the agent is balancing the pole for the whole permitted episode.

The raw curve will remain noisy because:

- exploration still occasionally chooses random actions;
- replay batches are random;
- small action differences can produce very different future trajectories;
- the policy changes the data that it subsequently encounters.

The 100-episode moving average shows the trend more clearly than individual
episodes.

## 14. The hyperparameters in the notebook

| Name | Value | What it controls |
|---|---:|---|
| `BATCH_SIZE` | 128 | Transitions used in one policy update |
| `GAMMA` | 0.99 | Importance of future reward |
| `EPS_START` | 0.9 | Initial probability of a random action |
| `EPS_END` | 0.01 | Long-run probability of a random action |
| `EPS_DECAY` | 2,500 | How slowly exploration decreases |
| `TAU` | 0.005 | How quickly the target follows the policy |
| `LR` | $3\times10^{-4}$ | AdamW learning rate |
| Memory capacity | 10,000 | Maximum stored transitions |

$3\times10^{-4}$ is read “three times ten to the negative four,” which is
`0.0003`.

These values interact. For example, very fast epsilon decay can prevent enough
exploration, while a very large learning rate can make Q-values unstable. There
is no guarantee that every random training run will learn at the same speed.

## 15. The shortest useful summary

The policy network predicts:

$$
\hat q_t=Q_\theta(s_t,a_t)
$$

The target network helps construct:

$$
y_t=r_t+\gamma\max_{a'}Q_{\bar\theta}(s_{t+1},a')
$$

The loss asks the policy prediction to move toward the target:

$$
L=\operatorname{Huber}(\hat q_t,y_t)
$$

The target network then moves a small amount toward the policy network:

$$
\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta
$$

In plain English:

> Predict the value of the action taken. Build a better target from the reward
> and the next state. Adjust the policy toward that target. Slowly copy the
> improved policy into the target network. Repeat with many remembered
> experiences.

## 16. Questions to test your understanding

1. Why does `policy_net` output two values instead of one?
2. Why does `gather` select only the action that was actually taken?
3. Why is the future value zero for a terminal state?
4. Why is `target_net` inside `torch.no_grad()`?
5. What would happen if epsilon stayed at `0.9` forever?
6. Why can an action be bad even though its immediate CartPole reward is `+1`?

<details>
<summary>Answers</summary>

1. There are two possible actions, and the network estimates the value of each.
2. The transition only reveals the outcome of the action that was taken.
3. No rewards occur after a true terminal state.
4. The target should be treated as fixed; only `policy_net` is optimized by the
   loss.
5. The agent would continue choosing random actions 90% of the time, preventing
   the learned policy from controlling most interactions.
6. It may lead to termination, so it receives no future rewards after the
   immediate `+1`.

</details>
