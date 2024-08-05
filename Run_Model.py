from Functions import *
from tqdm import tqdm
from datetime import datetime

if __name__ == '__main__':
    t_start = datetime.now()

    n_reps = 3
    K = 1.7
    k_n = 0.03

    mod_params_forward ={'k+':K*k_n, 'k-':k_n, 'tf':1000, 'dt':1, 'n': 200}
    mod_params_reverse ={'k+':0, 'k-':k_n, 'tf':200, 'dt':1, 'n': 200}

    model_forward = Model(mod_params_forward)
    model_reverse = Model(mod_params_reverse)

    Model_Bank = np.empty([n_reps, 2], dtype=object)

    for c in tqdm(range(n_reps)):

        forward_data = model_forward.run_model()
        reverse_data = model_reverse.run_model(initial=forward_data.final_time())

        reverse_data.fit_exp()

        Model_Bank[c, 0] = forward_data
        Model_Bank[c, 1] = reverse_data

        if forward_data.get_edge():
           print('Reached edge of domain')

    np.save('Model_Bank.npy', Model_Bank, allow_pickle=True)

    t_end=datetime.now()
    print('Time:',t_end-t_start, ' Start:', t_start, ' Finish:', t_end)

