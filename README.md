# 495Project

This project consists of a framework for defining robot structures using L-systems in a TaiChi MPM simulation, as well as an evolutionary algorithm for generating L-systems that produce robots with favorable traits, such as good locomotion ability in the simulation. L-systems are used to define both the structure and actuation controls of the robots. Avoiding the use of a separate gradient descent or reinforcement learning step to determine the robots' actuation controls allows the evolutionary algorithm to be much faster.

## L-system Framework

The common set of symbols used for the L-systems is shown in the table below, along with their interpretations. When interpreting these symbols, the rectangular body segments of the robot are drawn in a "turtle graphics" manner such that drawing a segment advances the cursor position to the end of the segment. Note that each symbol has numerical parameters that control how it is interpreted.

| Symbol | Interpretation |
|--------|---------------|
| $+\theta()$ | Turn counterclockwise by $\theta$ |
| $-\theta()$ | Turn clockwise by $\theta$ |
| $[$ | Save the current position and angle |
| $]$ | Restore the last-saved position and angle |
| $\{$ | Begin a block of symbols whose commands will be repeated |
| $s\}()$ | End a block of symbols whose commands will be repeated, and repeat them $s$ times |
| $F(l, w, a, \omega, \phi)$ | Create an actuated rectangle of dimensions $l \times w$ at the current position; set the actuation to be a sine wave of amplitude $a$, frequency $\omega$, and phase shift $\phi$ |
| $G(l, w)$ | Create an unactuated rectangle of dimensions $l \times w$ at the current position |

This L-system interpretation considers a production rule to be a special type of symbol that, like other symbols, can have associated input parameters. Each production rule additionally contains a pattern, which is the sequence of symbols that it transforms into when the L-system is iterated, and a matrix of weights that maps its inputs to the inputs of the symbols in the pattern.

For instance, consider a production rule $P_1(x, y)$ with pattern:

$$ G + P_1 [ ] $$

and matrix:

$$
\begin{bmatrix}
2 & 0 & 0 \\
0 & 3 & 0 \\
0 & 1 & 0 \\
1 & 0 & 1 \\
1 & 1 & 0
\end{bmatrix}
$$

The matrix maps the vector $(x, y, 1)^T$ to:

$$
\begin{bmatrix}
2x \\
3y \\
y \\
x+1 \\
y
\end{bmatrix}
$$

So if $P_1(x, y)$ appears in the axiom, when the L-system is iterated, it is replaced by $G(2x, 3y)$ and $P_1(x+1, y)$. This treatment of production rules as special symbols differs from the standard definition of an L-system, but it is equivalent.

## Evolutionary Algorithm

An initial generation of L-systems is created randomly. Each L-system's axiom consists of randomly selected symbols and random parameters. Production rules with randomly chosen patterns and random weights are also created. To prevent the creation of L-systems that produce robots that are impossible to simulate, some bounds on this random generation process are enforced. Axioms and patterns have minimum and maximum lengths, and each L-system must contain at least one production rule that refers to itself in its pattern and appears in the starting axiom. When determining the values of the axiom symbols' parameters, each parameter is associated with a particular Gaussian distribution that results in reasonable starting values. This prevents generation of robots with extremely large or small body segments, or actuation parameters that will immediately cause the particles to break apart. Also, when the L-systems are interpreted to produce robots in the MPM simulation, a cap of 3000 particles is enforced. Construction of a robot halts and its motion is simulated as-is once this cap is reached.

From each generation, a certain proportion of robots that score the best on some fitness measure are mutated to produce the next generation. L-systems can mutate in the following ways:

Symbols can be added, deleted, or replaced in the axiom or in the pattern of a production.

The input parameters of a symbol in the axiom can be modified.

A production's weights can be modified.

The number of iterations can increase or decrease.

Currently, the fitness measure is the distance traveled horizontally by the robot over the course of the simulation. This is scored based on the displacement of the robot's leftmost point, to prevent long robots that simply collapse or unfold in the horizontal direction.

## Results

The evolutionary algorithm was tested using a generation size of 50 robots, mutating the top 5 from each generation to produce the next, and creating 50 generations total. Across multiple runs, the algorithm reliably arrives at ball-shaped structures that roll very well, such as the one shown below.

(Insert GIF here)

The plot below shows the top fitness score and the fitness score cutoff for the top decile for each generation in one run of the algorithm. Since the fitness measure uses the displacement of the robot's leftmost point and the screen has width 1, scores of 0.8 and higher are very difficult to improve upon while maintaining a robot of a reasonable size.

L-systems are highly sensitive to small mutations. The alteration of even a single symbol can silence the generation of large parts of a robot's structure and greatly reduce its fitness. The significant regressions in the fitness score, especially for the top decile, on the plot above are therefore to be expected.

An optional penalty was later added to reduce a robot's fitness based on the approximate number of rotations it makes over the course of the simulation. On every time step, the rotation angle of each particle about the structure's center of mass relative to the previous time step is computed, an average is taken over all particles. The average rotation angles from each time step are summed at the end of the simulation, multiplied by a weighting constant, and deducted from the fitness score. This successfully encouraged the development of robots that move by walking or bouncing, one of which is shown below.

(Insert image here)

## Setup of the Codebase

The definition of the L-system framework, symbols, and mutation rules is in `LSystem.py`. The evolutionary algorithm is contained in `evolve.py`. When run, it creates a folder called `"structures"` that saves each L-system as a pickled `.json` file and includes screenshots of each robot. The file `loadStructure.py` loads a single L-system from a `.json` file, and displays the MPM simulation of the resulting robot. It can also be used to export a `.mp4` video of the simulation. The parsing of L-systems to create robots and the MPM simulation is done in `mpm.py`.
