import jax
from functools import partial


@partial(jax.vmap, in_axes=(0, 0, None), out_axes=-1)
def _create_mistakes(key, p, T):
    mistakes = jax.random.bernoulli(key, p=p, shape=(T,))
    return mistakes


def bern_oracle_beta_forecasters(key, n_experts, n_timesteps, a_fcst=2, b_fsct=7, p_oracle=0.5):
    """
    Create a set of forecasters whose probability of being correct is
    distributed according to a Beta(a_fcst, b_fcst) distribution.
    The oracle is a Bernoulli random variable with p=p_oracle.
    """
    key_oracle, key_noise, key_errs = jax.random.split(key, 3)
    keys_noise = jax.random.split(key_noise, n_experts)

    oracle = jax.random.bernoulli(key_oracle, p=p_oracle, shape=(n_timesteps,))
    ps_mistakes = jax.random.beta(key_errs, a=a_fcst, b=b_fsct, shape=(n_experts,))

    mistakes = _create_mistakes(keys_noise, ps_mistakes, n_timesteps)
    experts = (mistakes ^ oracle[:, None]).astype(float)

    return oracle, experts