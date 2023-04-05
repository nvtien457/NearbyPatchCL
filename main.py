from tools import Logger

logger = Logger(tensorboard=True, matplotlib=True, log_dir='../logs')

best_loss = logger.load_event('events.out.tfevents.1678642218.selab4')

print('Best loss:', best_loss)