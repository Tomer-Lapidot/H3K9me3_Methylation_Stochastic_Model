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

