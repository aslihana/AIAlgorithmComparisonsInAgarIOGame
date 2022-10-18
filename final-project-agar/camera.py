import config as conf


class Camera():
    """viewport for the current player"""

    def __init__(self, x, y, player_radius):
        self.x_pos = x
        self.y_pos = y
        self.player_radius = player_radius

    def pan(self, x, y):
        self.x_pos = conf.SCREEN_WIDTH / 2 - x
        self.y_pos = conf.SCREEN_HEIGHT / 2 - y

    def move_left(self, vel):
        """pan camera to the left"""
        left_bound = conf.SCREEN_WIDTH / 2 - self.player_radius
        self.x_pos = min(self.x_pos + vel, left_bound)

    def move_right(self, vel):
        """pan camera to the right"""
        right_bound = self.player_radius - conf.SCREEN_WIDTH / 2
        self.x_pos = max(self.x_pos - vel, right_bound)

    def move_up(self, vel):
        """pan camera up"""
        top_bound = conf.SCREEN_HEIGHT / 2 - self.player_radius
        self.y_pos = min(self.y_pos + vel, top_bound)

    def move_down(self, vel):
        """pan camera down"""
        bottom_bound = self.player_radius - conf.SCREEN_HEIGHT / 2
        self.y_pos = max(self.y_pos - vel, bottom_bound)
    
    def get_pos(self):
        """get position tuple"""
        return (self.x_pos, self.y_pos)