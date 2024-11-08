import random

import numpy as np
import pandas as pd


def simulate_profile_follicles_ml(profile, growth_model, chance_of_growth_model, features):
    np.random.seed(0)
    random.seed(0)

    simulated_follicles = []
    for follicle in profile.follicles:
        X = {'follicle': follicle, 'day': features['day'], 'age': features['age'],'afc': features['afc'],
             'amh': features['amh'],
             'weight': features['weight']
             }
        X = pd.DataFrame(X, index=[0])
        # print(X)
        chance_of_growth = chance_of_growth_model.predict(X)
        if chance_of_growth > 0.5:
            growth_in_perc = growth_model.predict(X)
            simulated_follicles.append(follicle * growth_in_perc)
        else:
            simulated_follicles.append(follicle)
    return simulated_follicles


