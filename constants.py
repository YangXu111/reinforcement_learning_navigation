CONSTANTS = {
    'hidden_layer_size': 128,
    'random_seed': 2,
    'learning_rate': 0.01,
    'tau': 0.1,
    'gamma': 0.99,
    'memory_size': 1000,
    'update_interval': 4,
    'sample_size': 64,
    'num_episodes': 6000,
    'epsilon_begin': 1,
    'epsilon_stable': 0.01
}
CONSTANTS.update({
    'epsilon_decay': (CONSTANTS['epsilon_stable']/CONSTANTS['epsilon_begin']) ** (1/CONSTANTS['num_episodes'])
})


PRIORITIZED_REPLAY_CONSTANTS = CONSTANTS.copy()
PRIORITIZED_REPLAY_CONSTANTS.update({
    'small_const': 0.01,
    'alpha': 0.6,
    'beta_begin': 0.4,
    'beta_increase': 0.0003,
    'beta_stable': 1
 })