import pygame
import math
from utils import scale_image, blit_rotate_center
import neat
import graphviz
import matplotlib.pyplot as plt
from neat.graphs import feed_forward_layers
import visualize
pygame.font.init()


# Load configuration file
config_path = "./neat-config.txt"
config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

scaling_factor = 0.85

""" Map 1 """
# GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5 * scaling_factor)
# TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9 * scaling_factor)
# TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9 * scaling_factor)
# FINISH = scale_image(pygame.image.load("imgs/finish.png"), scaling_factor)
# FINISH_MASK = pygame.mask.from_surface(FINISH)
# RACE_CAR = scale_image(pygame.image.load("imgs/race-car2.png"), 0.07 * scaling_factor)
# images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
#           (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
# START_POSITION = (170 * scaling_factor, 200 * scaling_factor)
# START_ANGLE = 0
# FINISH_POSITION = (130 * scaling_factor, 250 * scaling_factor)


""" Map 2 """
TRACK = scale_image(pygame.image.load("imgs/track.jpg"), 0.718 * scaling_factor)
TRACK_BORDER = scale_image(pygame.image.load("imgs/border.png"), 2 * scaling_factor)
RACE_CAR = scale_image(pygame.image.load("imgs/race-car.png"), 0.11 * scaling_factor)
images = [(TRACK, (0, 0))]
START_POSITION = (900 * scaling_factor, 50 * scaling_factor)
START_ANGLE = 90
FINISH_POSITION = (130 * scaling_factor, 250 * scaling_factor)


TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS = 100

class AbstractCar:
    def __init__(self, max_vel, rotation_vel, img):
        self.img = img
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = START_ANGLE
        self.x, self.y = self.START_POS
        self.acceleration = 1
        self.height = self.img.get_height()
        self.width = self.img.get_width()
        self.hitbox, self.hitbox_radius = self.create_hitbox()

    def create_hitbox(self):
        """ Create a circle hitbox for the car. """
        # Define the circle parameters
        circle_radius = self.width / 2
        circle_center = (circle_radius, circle_radius)
        circle_color = (255, 255, 255, 255)  # White color, fully opaque
        surface_size = (self.width, self.width)

        # Create a surface for the circle
        circle_surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        circle_surface.fill((0,0,0,0))  # Fill with transparent color

        # Draw the circle on the surface
        pygame.draw.circle(circle_surface, circle_color, circle_center, circle_radius)

        # Create a mask from the surface
        return pygame.mask.from_surface(circle_surface), circle_radius

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)

    def move_forward(self):
        self.vel = self.max_vel
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask):

        x = self.x + (self.width / 2) - self.hitbox_radius
        y = self.y + (self.height / 2) - self.hitbox_radius

        poi = mask.overlap(self.hitbox, (int(x), int(y)))

        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = START_ANGLE
        self.vel = 0


class PlayerCar(AbstractCar):
    START_POS = START_POSITION

    def __init__(self, max_vel, rotation_vel, id):
        super().__init__(max_vel, rotation_vel, RACE_CAR)
        self.sensor_length = 150  # Length of the sensor rays
        self.num_sensors = 5  # Number of sensors (rays)
        self.sensors = []
        self.sensor_data = []

    def update_sensors(self, obstacles):
        self.sensors.clear()
        self.sensor_data.clear()

        # Calculate the center of the car
        car_center_x = self.x + self.width / 2
        car_center_y = self.y + self.height / 2

        # Calculate the front of the car
        front_x = car_center_x + (self.height / 2) * math.cos(math.radians(self.angle+90))
        front_y = car_center_y - (self.height / 2) * math.sin(math.radians(self.angle+90))

        # Define sensor angles relative to the car's angle
        angles = [-45, -22.5, 0, 22.5, 45]  # Angles for the sensors in degrees

        for angle_offset in angles:
            # Calculate the sensor angle
            sensor_angle = self.angle + angle_offset + 90
            sensor_angle_rad = math.radians(sensor_angle)

            # Calculate the end point of the sensor ray starting from the front of the car
            end_x = front_x + self.sensor_length * math.cos(sensor_angle_rad)
            end_y = front_y - self.sensor_length * math.sin(sensor_angle_rad)  # Note the negative sign for y

            closest_distance = self.sensor_length
            closest_point = (end_x, end_y)

            # Check for collision with each obstacle
            for obstacle in obstacles:
                intersect_point, distance = self.ray_intersect((front_x, front_y), (end_x, end_y), obstacle)
                if intersect_point and distance < closest_distance:
                    closest_distance = distance
                    closest_point = intersect_point

            self.sensors.append((front_x, front_y, closest_point[0], closest_point[1]))
            self.sensor_data.append(closest_distance / self.sensor_length)  # Normalize distance

    def ray_intersect(self, start, end, obstacle_mask):
        x1, y1 = start
        x2, y2 = end

        for i in range(self.sensor_length):
            u = i / self.sensor_length
            x = int(x1 + u * (x2 - x1))
            y = int(y1 + u * (y2 - y1))

            try:
                if obstacle_mask.get_at((x, y)):
                    return (x, y), math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            except:
                pass

        return None, self.sensor_length

    def draw_sensors(self, win):
        # Draw the sensors for visualization
        for sensor in self.sensors:
            pygame.draw.line(win, (255, 0, 0), (sensor[0], sensor[1]), (sensor[2], sensor[3]), 3)
    
    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()


def draw(win, images, player_car: PlayerCar):
    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)

    # Update and draw sensors
    player_car.update_sensors([TRACK_BORDER_MASK])
    player_car.draw_sensors(WIN)

def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a]:
        player_car.rotate(left=True)
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved = True
        player_car.move_backward()

    if not moved:
        player_car.reduce_speed()


def handle_collision(player_car: PlayerCar):
    # Check if player hitts the border
    if player_car.collide(TRACK_BORDER_MASK) != None:
        player_car.reset()

	# Check if the player has finished
    # player_finish_poi_collide = player_car.collide(
    #     FINISH_MASK, *FINISH_POSITION)
    # if player_finish_poi_collide != None:
    #     if player_finish_poi_collide[1] == 0:
    #         player_car.reset()
    #     else:
    #         player_car.reset()


def visualize_genome(genome, config, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False, node_colors=None, fmt='png'):
    dot = graphviz.Digraph(format=fmt)
    
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    
    if node_names is None:
        # node_names = {}
        node_names = {-5: 'L ', -4: 'LC', -3: 'C ', -2: 'RC', -1: 'R ', 0: 'L ', 1: 'R '}

    if node_colors is None:
        node_colors = {}
    
    # Input nodes
    for i, n in enumerate(inputs):
        name = node_names.get(n, str(n))
        dot.node(name, _attributes={
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': 'lightgray',
            'pos': f"-1,{len(inputs) - i}!",
            'size': '1',
        })

    # Output nodes
    for i, n in enumerate(outputs):
        name = node_names.get(n, str(n))
        dot.node(name, _attributes={
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': node_colors.get(n, 'lightblue'),
            'pos': f"10,{len(outputs) - i}!",
        })

    # Hidden and bias nodes
    hidden_nodes = set(genome.nodes.keys()) - inputs - outputs
    layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, genome.connections)

    for i, layer in enumerate(layers):
        for n in layer:
            name = node_names.get(n, str(n))
            if n in hidden_nodes:
                dot.node(name, _attributes={
                    'shape': 'circle',
                    'style': 'filled',
                    'fillcolor': node_colors.get(n, 'white'),
                    'pos': f"0,{len(layer) - i}!"
                })

    # Connections
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_node, output_node = cg.key
            style = 'solid' if cg.enabled else 'dotted'
            dot.edge(node_names.get(input_node, str(input_node)),
                     node_names.get(output_node, str(output_node)),
                     _attributes={'style': style})

    # Graph attributes for styling
    dot.attr(overlap='false', splines='true', rankdir='LR')

    if filename is not None:
        dot.render(filename, view=view)

    return dot

def visualize_gen(config, genome):
    node_names = {-5: 'left', -4: 'left-center', -3: 'center', -2: 'right-center', -1: 'right', 0: 'left', 1: 'right'}

    visualize.draw_net(config, genome, True, node_names=node_names, prune_unused=False)

    # visualize.plot_stats(stats, ylog=False, view=True)

    # visualize.plot_species(stats, view=True)


def eval_genomes(genomes, config):
    # Initialize the game window
    clock = pygame.time.Clock()
    cars = []

    best_genome = None
    best_fitness = float('-inf')

    # Create a PlayerCar for each genome
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = PlayerCar(10, 12, genome_id)  # Create a car instance
        genome.fitness = 0
        cars.append((car, net, genome))  # Store car, its network, and genome

    run = True
    while run and len(cars) > 0:
        clock.tick(FPS)
        WIN.fill((255, 255, 255))  # Clear screen with white background

        # Draw background images
        for img, pos in images:
            WIN.blit(img, pos)

        # Update and draw each car
        for car, net, genome in cars:
            if car.collide(TRACK_BORDER_MASK) is not None:
                # genome.fitness -= 1  # Penalize for collision
                cars.remove((car, net, genome))  # Remove crashed car
            else:
                # Update sensors and feed sensor data to the neural network
                car.update_sensors([TRACK_BORDER_MASK])
                output = net.activate(car.sensor_data)

                # Control car based on network output
                steerLeft, steerRight = output[0], output[1]
                if steerRight > 0.5 or steerRight > 0.5:
                    if steerRight > steerLeft:
                        car.rotate(right=True)
                    else:
                        car.rotate(left=True)

                car.move_forward()

                # car.draw_sensors(WIN)

                car.draw(WIN)

                # Increase fitness for moving forward
                genome.fitness += 0.1

                # Track the best genome
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    if best_genome is not None:
        # Visualize the best genome's neural network
        visualize_genome(best_genome, config, filename="best_genome")
        # visualize_gen(config, best_genome)


def test_mode():
    run = True
    clock = pygame.time.Clock()

    player_car = PlayerCar(4, 4, 0)

    while run:
        clock.tick(FPS)

        draw(WIN, images, player_car)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        move_player(player_car)

        handle_collision(player_car)

        pygame.display.update()

    pygame.quit()

# test_mode()

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

winner = p.run(eval_genomes, n=1000)

print(f'Best genome:\n{winner}')

