# H3K9me3_Methylation_Stochastic_Model
Stochastic model on H3K9me3 methylation. Propagation of histone activation and deactivation dynamics.

This is the model used in the paper:
Different H3K9me3 heterochromatin maintenance dynamics govern distinct gene programs and repeats in pluripotent cells
(Nature Cell Biology, 2024)

The model presented here is based on a discrete one-dimensional lattice of nucleosomes on a chromatin strand. The simulated lattice domain spans over 1000 nucleosomes, where each nucleosome can have one of two states, unmodified (with a value of 0), or modified (with a value of 1). The model consists of three processes that govern the state of a nucleosome.

Nucleation: A process that can only occur at the central nucleosome of the domain, and only if the center nucleosome is in an unmodified state. The process is governed by the rate constant k+.

Propagation: A process that can turn a nucleosome from an unmodified to a modified state under the condition that an adjacent nucleosome is in a modified state. Like nucleation, the process is governed by the rate constant k+.

Turnover: A process that can turn a nucleosome from a modified back to an unmodified state. Unlike propagation, turnover can occur in any nucleosome that is in a modified state regardless of the state of adjacent nucleosomes. This process is governed by the rate constant k-.

Each of the three processes proceeds by probabilistic description. A process proceeds based its probability of occurrence p, given by the following equations for nucleation and propagation,

p_nuc=k_+ ∆t
p_prg=k_+ ∆t

And for turnover,

p_trn=k_- ∆t


Every time iteration, the probability of a process occurring in a nucleosome is compared against a number generated by a pseudo-random uniform distribution between 0 and 1.  For simplicity, we chose to maintain ∆t=1, such that the value of k+ and k- reflect the probability of the processes. In addition, we found that it was convenient to introduce K, the ratio between k+ and k- such that,
K=k_+/k_- 

The boundary nucleosomes of the lattice were expected, but not forced, to remain in an unmodified condition to describe a fully independent propagation process. As a result, simulations were limited to rate constant values that did not result in propagation that reached the boundary of the lattice. A k+ and k- combination was allow only if at least 99% of simulation repeats did not reach the lattice boundaries. Furthermore, a large lattice domain (1000 nucleosomes) was used to accommodate a wide range of k- and k+ values.
To simulate the nucleosome methylation process, each simulation consists of two stages:

1. Forward stage: With the initial condition that all nucleosomes in the lattice are in an unmodified state. Nucleation, propagation, and turnover are allowed to proceed until the number of modified nucleosomes achieves a steady state. Here we modified the values of k- between 0.0335 and 0.184, and K between 1 and 1.6, the value of k+ was calculated accordingly. This stage was allowed to proceed for 15,000-time steps, which provided enough time for the number of nucleosomes to achieve steady state even at the highest K values without reaching the lattice boundaries.

2. Reverse stage: After the forward time steps, the value of k+ was set to 0, allowing only turnover to proceed. The configuration of lattice at the last time step of the forward stage served as the initial condition for the reverse stage. The lattice was allowed to return to a state where all nucleosomes were unmodified, for which 500 additional time steps were sufficient for all K values.

The reverse methylation simulation was fitted to an exponential decay curve with two characteristic degrees of freedom,

N=N_0  exp⁡(-a∆t)

Where N_0 is the number of modified nucleosomes at the end of the forward stage (and therefore the beginning of the reverse stage), and a is the decay constant of the exponential curve. Fitting was done by linearization of exponential curve and performing a linear regression against the reverse stage simulation results. The simulation was repeated 2000 times for every k- and K pair, and curve parameters N_0 and a were fitted for every simulation repeat.

