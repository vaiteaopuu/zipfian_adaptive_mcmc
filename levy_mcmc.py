from numpy.random import normal, standard_cauchy, uniform
from numpy import exp, array, linspace, zeros, arange, mean, argmin
from scipy.stats import pearsonr, cauchy
from multiprocessing import Pool

from matplotlib import pyplot as plt


H_STEP = 0.0005
DELTAT = 200.0


def update_bias(histo, dist, x_dist, nb_el):
    for i in range(nb_el):
        histo[i, :] += H_STEP * exp(-(x_dist - dist[i])**2/0.005) * exp(-histo[i, :]/DELTAT)
    return histo


def prob_func(x):
    return exp(-(x**4 - 200 * x**2).sum()/100)


def mcmc(nb_steps, nb_el, mut_prob):
    old_conf = uniform(-15, 15, size=nb_el)
    old_prob = prob_func(old_conf)
    traj = []

    x_dist = linspace(-20, 20, 400)
    histo = zeros((nb_el, 400))

    dist_o = [argmin((x_dist - el)**2) for el in old_conf]
    old_b_prob = prob_func(array([histo[si, i] for si, i in enumerate(dist_o)]))

    acc = 0
    for step in range(nb_steps):
        new_conf = old_conf + mut_prob(nb_el)
        new_prob = prob_func(new_conf)
        dist_n = [argmin((x_dist - el)**2) for el in new_conf]
        new_b_prob = prob_func(array([histo[si, i] for si, i in enumerate(dist_n)]))

        # if new_prob > old_prob or new_prob/old_prob >= uniform(0, 1):
        if new_prob * old_b_prob > old_prob * new_b_prob or (new_prob * old_b_prob)/(old_prob * new_b_prob) >= uniform(0, 1):
            old_conf = new_conf
            old_prob = new_prob
            acc += 1
        traj += [old_conf]

        # if step % 10 == 0:
        update_bias(histo, old_conf, x_dist, nb_el)
        dist_o = [argmin((x_dist - el)**2) for el in old_conf]
        old_b_prob = prob_func(array([histo[si, i] for si, i in enumerate(dist_o)]))
    return array(traj), histo, x_dist


def run_test(args):
    sc_val, cauchy_b, nb_steps, nb_el = args
    if cauchy_b:
        cauchy_gen = cauchy(scale=sc_val)
        cauchy_step = lambda size: cauchy_gen.rvs(size=size)
        traj, histo, x_dist = mcmc(nb_steps, nb_el, cauchy_step)
    else:
        normal_step = lambda size: normal(size=size, scale=sc_val)
        traj, histo, x_dist = mcmc(nb_steps, nb_el, normal_step)

    true_prob = array([prob_func(el) for el in x_dist])
    true_prob /= sum(true_prob)
    pr_val = []
    for i in range(nb_el):
        pred_prob = exp(histo[i, :])/sum(exp(histo[i, :]))
        pr_val += [pearsonr(pred_prob, true_prob)[0]]
    return sc_val, mean(pr_val)

    


def main():
    nb_el = 4
    # optim = True
    optim = False

    pool = Pool(4)

    cauchy_b = True
    scale_val = 1.0

    # cauchy_b = False
    # scale_val = 10

    nb_steps = 100000

    if optim:
        if cauchy_b:
            res = pool.map(run_test, [(sc_val, cauchy_b, nb_steps, nb_el) for sc_val in arange(5, 15, 0.1)])
        else:
            res = pool.map(run_test, [(sc_val, cauchy_b, nb_steps, nb_el) for sc_val in arange(5, 20, 0.1)])
        for sc_val, cor in res:
            print(sc_val, cor)
        print("MAX:", max(res, key=lambda el: el[1]))

    else:
        if cauchy_b:
            cauchy_gen = cauchy(scale=scale_val)
            cauchy_step = lambda size: cauchy_gen.rvs(size=size)
            traj, histo, x_dist = mcmc(nb_steps, nb_el, cauchy_step)
        else:
            normal_step = lambda size: normal(size=size, scale=scale_val)
            traj, histo, x_dist = mcmc(nb_steps, nb_el, normal_step)

        fig = plt.figure(1)
        dens_f = [fig.add_subplot(int((nb_el+1) * 100 + 11 + i)) for i in range(nb_el+1)]
        true_prob = array([prob_func(el) for el in x_dist])
        true_prob /= sum(true_prob)

        for i in range(nb_el):
            print(pearsonr(pred_prob, true_prob))
            pred_prob = exp(histo[i, :])/sum(exp(histo[i, :]))
            dens_f[i].plot(x_dist, pred_prob)
            dens_f[i].set_ylabel("density")
            dens_f[i].set_xlim([-20, 20])
            dens_f[i].tick_params(axis="x", labelbottom=False, size=0)
            # dens_f[i].hist(traj[:, i], bins=40)

        dens_f[-1].plot(x_dist, true_prob)
        dens_f[-1].set_xlim([-20, 20])
        dens_f[-1].set_xlabel("Sample value")
        dens_f[-1].set_ylabel("density")

        # hist_f = fig.add_subplot(142)
        # hist_f.hist(traj, bins=100)
        # # plt.savefig("cauchy.png")

        plt.show()


if __name__ == '__main__':
    main()
