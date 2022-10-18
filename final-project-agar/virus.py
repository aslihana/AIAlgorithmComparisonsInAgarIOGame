class Virus():
    ID_counter = 0

    def __init__(self, x, y, r, mass):
        self.x_pos = x
        self.y_pos = y
        self.radius = r
        self.mass = mass
        self.is_alive = True

        # give unique IDs to objects for debug purposes
        self.id = Virus.ID_counter
        Virus.ID_counter += 1

    def get_pos(self):
        return (self.x_pos, self.y_pos)
