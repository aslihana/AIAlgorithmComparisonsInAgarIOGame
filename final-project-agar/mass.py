import config as conf
import utils


class Mass():
    ID_counter = 0

    def __init__(self, x, y, color, angle, parentCellRadius):
        self.x_pos = x
        self.y_pos = y
        self.mass = conf.MASS_MASS
        self.radius = utils.mass_to_radius(self.mass)
        self.color = color
        self.angle = angle

        # give unique IDs to objects for debug purposes
        self.id = Mass.ID_counter
        Mass.ID_counter += 1

        self.velocity = self.radius * 2
        self.acceleration = self.radius / 4

        # Move out from the parent's radius
        utils.move_object(self, self.angle, parentCellRadius + self.radius)

    def move(self):
        if not self.velocity:
            return

        utils.move_object(self, self.angle, self.velocity)
        self.velocity = self.velocity - self.acceleration

        if self.velocity <= 0:
            self.velocity = 0
            self.acceleration = 0
            self.angle = None

    def is_moving(self):
        return self.angle is not None

    def get_pos(self):
        return (self.x_pos, self.y_pos)
