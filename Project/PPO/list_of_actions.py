from pysc2.lib import actions

for action in obs.observation.available_actions:
    print(actions.FUNCTIONS[action])