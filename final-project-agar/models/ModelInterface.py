class ModelInterface:
    ID_counter = 0

    def __init__(self):
        self.id = ModelInterface.ID_counter
        self.done = False
        self.eval = False
        self.camera_follow = False
        ModelInterface.ID_counter += 1

    def get_action(self, state):
        """Given the current game state, determine what action the model will output"""
        raise NotImplementedError('Model get_action() is not implemented')

    def optimize(self):
        """Given reward received, optimize the model"""
        raise NotImplementedError('Model optimize() is not implemented')

    def remember(self, state, action, next_state, reward, done):
        """Update replay buffer with what model chose to do"""
        raise NotImplementedError('Model remember() is not implemented')
