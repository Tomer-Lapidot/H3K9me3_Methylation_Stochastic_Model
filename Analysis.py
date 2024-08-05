import matplotlib.pyplot as plt

from Functions import *
import sys

model_bank = np.load('Model_Bank.npy', allow_pickle=True)

# Choose which simulation interation to analyze
# Can iterate through range(len(model_bank)) to see all repeats
for run_idx in [0]:

    # Plot model run
    plt.figure(figsize=(12, 6), dpi=100)
    Plot.plot_run(model_bank[run_idx])
    plt.savefig('Model_Results.png')
    # plt.show()
    plt.clf()

    # Plot total nucleosomes over time for a model run
    plt.figure(figsize=(12, 6), dpi=100)
    Plot.plot_time_run(model_bank[run_idx])
    plt.savefig('Total_Nucleosomes_Over_Time.png')
    # plt.show()
    plt.clf()

    # Plot nucleosome deactivation fit
    plt.figure(figsize=(6, 6), dpi=100)
    model_bank[run_idx, 1].plot_fit()
    plt.savefig('Deactivation_Curve_Plot.png')
    # plt.show()
    plt.clf()

    # Plot mean of final times of activation model, here for different k+ values
    plt.figure(figsize=(6, 6), dpi=100)
    Plot.plot_final_time_model(model_bank, color='black', color_ribon='lightgrey')
    plt.savefig('Nucleosome_Final_Time_Repeat_Average.png')
    # plt.show()
    plt.clf()

    # Extract fitted parameters y0, a, and r2, for a bank of k+ and repeats
    param_data = np.zeros([len(model_bank), 3])
    for r in range(len(model_bank)):
        param_data[r] = model_bank[r, 1].get_params()


