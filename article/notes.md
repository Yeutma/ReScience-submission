## Abstract

The necessity of fine-tuning the meta-parameters ($\alpha$, $\beta$, $\gamma$) for successful completion of a reinforcement learning task represents a systematic hurdle. [@schweighofer2003meta] relates a biologically plausible, dynamic and adaptive learning algorithm for automatic online tuning of the meta-parameters, robust to environment changes in static and dynamic environments.  
We replicate Fig. 1 and Fig. 2, in which a Q-learning agent evolves in a discrete Markovian Decision Problem task inspired from [@tanaka2004prediction], for different meta-parameter initial conditions, number of states of the MDP task, and environment switches, as specified in [@schweighofer2003meta].  We also replicate Fig. 3, a non-linear continuous-time inverted pendulum control task taken from [@doya2002metalearning].  
After certain small corrections from the original manuscript, we find that the meta-learning algorithm's performance strongly diminishes with increasing states and fully depends upon the meta-parameter's initial conditions, contrary to what is claimed in [@schweighofer2003meta].  
Finally, we propose an alternative non-stochastic version of the meta-learning algorithm, which allows the agent's performance to robustly converge to a stable solution for increasing states, but also depends upon the meta-parameter's initial conditions.  

## Last section
Here, the values of these meta-parameters m will vary in a specific direction for increasing reward. The choice for each meta-parameters’ converging value (0 for $\alpha$, 1 for $\gamma$, +$\infty$ for $\beta$) was established according to the following principles: for increasing reward / task completion,
* the learning-rate $\alpha$ should decrease to refine the learning and stop the learning from spiraling out into infinity in case new learning needs to take place.  
* the inverse temperature $\beta$ increases from an initially low value to promote exploration and accumulation of environment knowledge to higher values for informed exploitation.  
* the discount factor $\gamma$ should increase to allow goal-oriented behavior on long distances once the world is mapped and reward discovered, but de-creased to allow disengaging from the known future reward now obsolete.  

## Examples

$\gamma$ sets the target Q-values, $\alpha$ the rate at which current Q-values converge to their target values, and $\beta$ which Q-values are updated.
* Albeit the obvious ($\alpha=0$ leads to no learning, $\beta=0$ explores), we observe multiple different cases which can be summarized into 5 main points:
    * Solution found since Intermediate states learned:
        * Initially high alpha decreases (even up to 0) / low alpha, initially low beta increases, high gamma / initially low for switch.
        Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°0
        * Tolerance to:
            * Alpha decreasing to 0: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°13
            * Varying Q-values = High alpha + varying gamma: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°2 / 75 / 79
            * Sudden increase in wrong-direction Q-values = high alpha (~1), low beta (~0), high gamma (~1) (updates other path): Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°17 / 23 / 27 / 65 / 73 / 79 / 82 / 107
            * Suboptimal performance albeit learning = low beta ()
    * Unstable solution:
        * Low gamma:
            * high alpha unlearns: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°59  68 / 131
            * even low alpha unlearns: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°99
        * High alpha: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°6 / 116 / 117 / 120
        * Matching wrong & right direction Q-values: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°108 / 126
    * No solution:
        * Beta high initially / alpha low then beta high: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°
        * Gamma low: Affine, Fig 1, beta 1 gamma 0.99, 4 states, n°38 / 99 / 109 / 126

* Multiple behaviors:
    * Intermediate Q-values learned = solution found. Works even for null alpha afterwards.
    * High alpha, varying gamma = varying Q-values. Generally leads to stable solution.
    * High alpha, low beta, high gamma = increase in wrong-direction Q-values since updated. Punctual or sustained.
    * Low beta = suboptimal performance albeit learning.
    * Low gamma with high / low alpha unlearns future / learns immediate present.
    * Beta initially high = agent can be stuck between two states whose Q-values have converged to wrong target values and are no longer updated.

* What algorithm must do:
    * Test large rewards multiple times for high Q-values, and then backpropagate Q-values to intermediate states while differentiating left vs right Q-values.
    As such, gamma must be high, alpha moderate (> 0 but low), beta high, and large Q-values explored.
    * How to explain Affine Beta, Fig 1, Beta 1 Gamma 0.99, 10 states, n°7? Hypothesis: Because it is stuck between two states whose Q-values have converged to wrong values and aren't updated.
    * As Q(RL) converges to RL, Q(-RL) converges to 0 (for gamma = 1).
    *
    Gamma low = blind Q-values, high = long-term Q-values.
    Alpha low = slow learning, high = fast learning.
    Beta low = all states explored, high = only high Q-values explored.
