import argparse


def compute_tau(kappa: float, beta: float) -> float:
    mises_maxprob = MisesProb(kappa=kappa, beta=beta)
    K = similarity.shape[-1]
    thresh_analitic = (
        1
        / kappa
        * (
            np.log(beta / (1 - beta))
            + np.log(K)
            + mises_maxprob.log_uniform_dencity
            - mises_maxprob.log_c
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kappa", metavar="N", type=float, nargs="+")
    parser.add_argument("beta", metavar="N", type=float, nargs="+")
    args = parser.parse_args()
    print(compute_tau(kappa=args.kappa, beta=args.beta))
