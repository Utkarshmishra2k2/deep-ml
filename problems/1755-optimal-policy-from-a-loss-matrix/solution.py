import numpy as np

def expected_losses(posterior, loss):
    return np.asarray(posterior) @ np.asarray(loss)

def bayes_policy(posterior, loss):
    return np.argmin(expected_losses(posterior, loss), axis=1)

def covid_table(prevalence, sens, spec, drug_cost,
                qaly_young=60.0, qaly_old=10.0):
    p_covid = {
        0: ((1 - sens) * prevalence) /
           ((1 - sens) * prevalence + spec * (1 - prevalence)),
        1: (sens * prevalence) /
           (sens * prevalence + (1 - spec) * (1 - prevalence)),
    }

    rows = []
    for test in (0, 1):
        for age in (0, 1):
            p = p_covid[test]
            cost_nothing = p * (qaly_young if age == 0 else qaly_old)
            cost_drugs = drug_cost
            action = int(cost_drugs < cost_nothing)
            rows.append([
                test, age, float(p), float(cost_nothing),
                float(cost_drugs), action
            ])

    return rows