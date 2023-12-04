# IntelligentEvolving
A Study of Evolution and Markov Blankets

There exists a video from 2020 by David Randall Miller where he programs creatures in an
artificial life program to simulate evolution. The behavior of these beings are determined by a genetic
code specifying a Markov Chain brain of the agents in the simulation. The Markov Chains in those
agents have a list of sensory and motor states connected by different weights to determine their
behavior in the environment. David’s genetic code mutates and causes his agents to exhibit a range of
complex behaviors through artificial natural selection.

This Agent Based Evolutionary Model here is inspired but quite different from the one in the
video. It uses a simplification in parameter specification (but not in parameter size) of the agent's brain
in order to carefully investigate a relationship between a changing environment and the creatures that
occupy it. Keeping spirit with David’s experiment, my agents also have a Markov Blanket delineating
the agents movement, sensory, and decision layers; however instead of careful specifications of possible
sensors, motors movements, and their relations to one another through transition probabilities, I
imbued each agent with their own Q-Learning paradigm so they could learn how to interact with their
own bodies using information from a limited set of sensorial inputs. The agent's body evolution is still
a randomly mutated process such that the actions determined by the agent brain are separated from
their physical phenotype. 

The Q-Learning paradigm I have set up gives agents abilities within their
lifetimes to master their environment and achieve long-standing reproduction goals through
maximization of relevant rewards, while leaving the purely physical body in the hands of natural
selection. The evolutionary dynamics of the “brain” and body become a corrective process in
themselves. Additionally, the Q-Matrix brain itself is passed down in a mutated form each generation
alongside a mutated body. This might seem like a problematic setup at first since specific knowledge is
never passed down genetically; however, the non deep Q-Learning framework is limited in cognition. It
is a simple tabular reinforcement learning algorithm that can be modeled in discrete states and
transition probabilities. Although it is capable of learning complex long term goals, it takes many steps
and a certain level of brute forcing to achieve that end. Epigenetics in real organisms also encodes
genomic expression on a massive level. The Q-Learning framework essentially represents a form of
epigenetic inheritance and a non-cognitive facility. We will later see that mutation rates do have a
profound effect on the agent population, even with Q-Matrix inheritance.

![population_food_regen_0 5](https://github.com/Zan-Huang/IntelligentEvolving/assets/10505540/104d4afb-bebc-4d42-a3f2-cfc664b3ba51)

![population_food_regen_0 75](https://github.com/Zan-Huang/IntelligentEvolving/assets/10505540/217cbe8b-678c-4937-9189-8d1818f63828)

![population_food_regen_0 25](https://github.com/Zan-Huang/IntelligentEvolving/assets/10505540/b3bcc0ac-7c16-4d0e-a13d-7724c4a2b0ef)


In every experiment, we find a reasonable exponential explosion in population and a massive
die off or curtailing within the first twenty generations. This is due to the initially unlimited availability
of food resources and its rapid depletion through mass consumption. The idea is that agents who can
adapt and evolve healthily given a certain mutation rate can more easily survive in a rapidly changing
low resource environment.

As expected, all agents died out before 120 generations for the 0.25 food regeneration rate
environment.

However, in the ideal 0.5 food regeneration environment, non mutation runs completely dies
off before 200 generations, while those with mutations survived past the simulation end at 450
generations. This result not only speaks to the intuitive importance of mutation in natural selection of
entire populations, but also to the fact that epigenetic or cognitive like planning is insufficient on its
own to succeed. Even for the extremely easy 0.75 food regeneration environment, mutation rates of 0
yielded significantly lower populations than those runs with higher mutation rates.
In the most difficult 0.25 food regeneration environment, the average population and survival
time was highest and longest for the highest mutation value of 0.5. For the optimal 0.5 food
regeneration environment, the highest rates of mutation (0.25 and 0.5) performed better than low
rates of 0.05 or 0.1.

The ability to mutate at a higher rate has a distinct advantage, even if is disruptive
to an agent's expected behavior, as the learning algorithm would have to operate on a changed body.
The intuitive finding is that a mutation is necessary for adaptation, even if agents are already
able to navigate their environment and transit refined information across generations. The slightly less
obvious finding is that higher mutation rates lead to more robustness in the survival of a population,
especially in environments with limited and changing resource distribution.
Even though its clear that non mutated agent populations would die off and that higher
mutation agents would perform better in changing environments, we must remember that these agents
also have an epigenetic-like learning algorithm that is capable of learning within their own lifetimes and
are able to transmit their behavior cross generation.

Thus, the really crucial finding, even though separate from the central question at hand, is that
placing a Markov Blanket in between the agents physical movement and an agents adaptive behavior
enforces the two aspects of this model to jointly optimize for a certain maximization even when the
two optimization strategies are completely separate and disconnected (Reinforcement Learning v.s.
Evolutionary Algorithm). More critically, that joint maximization even occurs on two distinct
timescales. The mutation and distribution of a Transition Matrix over several generations represents a
much larger timescale than the immediate Q learning optimization an agent makes during its own
lifetime, passing to its more temporally close descendants. Yet, the differing features still converge to
sustain life and complex phenomena.
