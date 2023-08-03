import jax


class PotentialForecaster:
    """
    Forecaster with potential-based forecast.
    """
    def __init__(self, potential, lossfn, n_experts):
        self.potential = potential 
        self.lossfn = lossfn
        self.gradp = jax.grad(potential)
        self.n_experts = n_experts
    
    def forecast(self, experts, regret_prev):
        """
        At time t, we make a prediction based on the
        gradient of the potential evaluated at the regret
        obtained at time t-1 and the prediction of each forecaster at time t.
        """
        weights = self.gradp(regret_prev)
        forecast = weights @ experts / weights.sum()
        return forecast
    
    def update_regret(self, forecast, experts, oracle, regret_prev):
        """
        Update vector of regrets for each expert
        """
        inst_regret = self.lossfn(forecast, oracle) - self.lossfn(experts, oracle)
        regret = regret_prev + inst_regret
        return regret
    
    def _update_step(self, regret, xs):
        """
        At time t, we make a prediction based on the
        regret (as defined by the potential) obtained at time
        t-1 and the prediction of each forecaster at time t.

        After making a new prediction, we update the regret.
        """
        experts, outcome = xs
        # Make a forecast based on the current experts and
        # the regret obtained at time t-1
        forecast = self.forecast(experts, regret)
        # Obtain vector of regrets for each expert
        regret = self.update_regret(forecast, experts, outcome, regret)

        out = {
            "forecast": forecast,
            "regret": regret
        }

        return regret, out
    
    def run(self, experts, oracle):
        ...