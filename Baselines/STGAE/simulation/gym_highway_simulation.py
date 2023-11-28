import os
import math
import numpy as np
import numpy.random as random
import json
import glob
import warnings
from datetime import datetime

# sim modules
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import gym
from gym.utils import colorize, seeding, EzPickle
import pyglet

# car model
import dynamic_car_model

FPS = 50

VIEWPORT_W = 2500
VIEWPORT_H = 400

ROAD_COLOR = [0.4, 0.4, 0.4]
N_DASHES = 50  # lane markings

CAR_COLORS = [(0.0, 0.0, 0.8), (0.8, 0.0, 0.0),
              (0.0, 0.8, 0.0), (0.0, 0.8, 0.8),
              (0.8, 0.8, 0.8), (0.0, 0.0, 0.0),
              (0.8, 0.0, 0.8), (0.8, 0.8, 0.0)]

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]

        # This check seems to implicitly make sure that we only look at wheels as the tiles
        # attribute is only set for wheels in car_dynamics.py.
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited[obj.car_id]:
                tile.road_visited[obj.car_id] = True
                self.env.tile_visited_count[obj.car_id] += 1

                # The reward is dampened on tiles that have been visited already.
                past_visitors = sum(tile.road_visited)-1
                reward_factor = 1 - (past_visitors / self.env.num_agents)
                self.env.reward[obj.car_id] += reward_factor * 1000.0/len(self.env.track)
        else:
            obj.tiles.remove(tile)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


class MultiCarCountryRoad(gym.Env):

    def __init__(self, num_agents=1, car_positioning="random", car_pos=None):
        EzPickle.__init__(self)
        self.seed()

        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.scale = 6.0  # affects how fast-paced the game is, forces should be adjusted as well
        self.viewer = None

        self.num_agents = num_agents
        self.cars = [None] * self.num_agents
        self.car_actions = [[] for _ in range(self.num_agents)]
        self.car_positioning = car_positioning
        if self.car_positioning in ["random", "aligned", "opposite"]:
            self.car_init = [None] * self.num_agents
        elif self.car_positioning in ["user"]:
            self.car_init = car_pos  # [x, y, angle, lane, street]
        else:
            warnings.warn("Unknown car positioning method: {}".format(self.car_positioning))
        self.car_colors = [None] * self.num_agents

        # recording
        self.cars_history = [[]] * self.num_agents
        self.label = []
        self.recording_time = []
        self.start_recording = False
        self.start_recording_id = 0
        self.end_recording = [False] * self.num_agents
        self.end_recording_id = -1

        # annotation
        normal_labels = ["side-by-side", "overtake left", "multi-overtake",
                         "following", "opposite drive", "else"]
        anomaly_labels = ["aggressive overtaking",
                          "pushing aside",
                          "right spreading",
                          "left spreading",
                          "tailgating",
                          "thwarting",
                          "leave road",
                          "staggering",
                          "skidding",
                          "wrong-way driving",
                          "aggressive reeving",
                          "else"]
        self.labels = {"normal": {n: k for n, k in enumerate(normal_labels)},
                       "abnormal": {n: k for n, k in enumerate(anomaly_labels)}}

        # agent executing the main action
        self.acting_agents = {n: k for n, k in enumerate(["red", "blue", "both"])}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_car_history(self, car_id):

        # Initialize car history
        wheels = []
        for wheel in self.cars[car_id].wheels:
            wheels.append(np.array(wheel.worldCenter))
        car_center = line_intersection((wheels[2], wheels[1]), (wheels[0], wheels[3]))
        car_angle = self.cars[car_id].hull.angle
        self.cars_history[car_id] = [(car_center + (car_angle,))]

    def get_init_position(self, lane_id):
        # define parameter distributions as gaussian [x, y, angle]
        x_mean = 20 / self.scale
        x_std = 5 / self.scale
        y_std = self.lane_width / 15
        angle_kappa = 400  # kappa=dispersion, the bigger the smaller the range

        param_distributions = {0: {'x': [x_mean, x_std],
                                   'y': [self.lower + self.lane_width / 2., y_std],
                                   'angle': [math.pi * (-0.5), angle_kappa]},
                               1: {'x': [x_mean, x_std],
                                   'y': [self.lower + self.lane_width * 1.5, y_std],
                                   'angle': [math.pi * (-0.5), angle_kappa]},
                               2: {'x': [self.W - x_mean, x_std],
                                   'y': [self.street_center + self.lane_width / 2., y_std],
                                   'angle': [math.pi * 0.5, angle_kappa]},
                               3: {'x': [self.W - x_mean, x_std],
                                   'y': [self.street_center + self.lane_width * 1.5, y_std],
                                   'angle': [math.pi * 0.5, angle_kappa]}}

        x = random.normal(param_distributions[lane_id]['x'][0], param_distributions[lane_id]['x'][1])
        y = random.normal(param_distributions[lane_id]['y'][0], param_distributions[lane_id]['y'][1])
        angle = random.vonmises(param_distributions[lane_id]['angle'][0], param_distributions[lane_id]['angle'][1])
        street_id = self.get_lane_to_street_map()[lane_id]

        return x, y, angle, lane_id, street_id

    def get_lane_to_street_map(self):

        return {0: 0, 1: 0, 2: 1, 3: 1}

    def init_car_positions(self):
        """ Initialize to position of the two cars on the two lane country road - following scheme:
        --------------------
                           3
        --------------------
                           2
        ####################
        1
        --------------------
        0
        --------------------
        """

        import itertools

        # generate combinations of start positions - avoid assiging two cars to the same position
        p1 = np.array([list(e) for e in itertools.combinations([0, 1, 2, 3], 2)])
        p2 = np.zeros(p1.shape)
        p2[:, [0, 1]] = p1[:, [1, 0]]
        all_start_positions = np.vstack((p1, p2))

        # non-uniform distribution >> s.t. aligned positions are not under-represented
        all_street_start_positions = np.array([self.get_lane_to_street_map()[l] for l in
                                              all_start_positions.flatten()]).reshape(all_start_positions.shape)
        aligend = np.where([np.all(arr == arr[0]) for arr in all_street_start_positions])[0]
        opposite = np.where([not np.all(arr == arr[0]) for arr in all_street_start_positions])[0]
        n_a = aligend.shape[0]
        n_o = opposite.shape[0]

        p_o = 1/(2*n_o)
        p_a = (n_o/n_a) * p_o

        probabilities = np.zeros(all_start_positions.shape[0])
        probabilities[opposite] = p_o
        probabilities[aligend] = p_a

        # filter possible start positions
        if self.car_positioning == "aligned":
            keep = aligend
            n_pos = all_start_positions[keep].shape[0]
            lane_ids = all_start_positions[keep][np.random.randint(n_pos)]
        elif self.car_positioning == "opposite":
            keep = opposite
            n_pos = all_start_positions[keep].shape[0]
            lane_ids = all_start_positions[keep][np.random.randint(n_pos)]
        else:
            keep = np.arange(all_street_start_positions.shape[0])
            n_pos = all_street_start_positions.shape[0]
            lane_ids = all_start_positions[np.random.choice(np.arange(n_pos), p=probabilities)]

        # sample from distribution
        car_start_positions = []
        for car_id, lane_id in enumerate(lane_ids):
            x, y, angle, lane_id, street_id = self.get_init_position(lane_id=lane_id)
            car_start_positions.append([x, y, angle, lane_id, street_id])

        return car_start_positions

    def _destroy(self):
        pass

    def reset(self, car_positioning="random"):

        # label
        self.label.append(-1)

        # timestep
        self.t = 0.0

        self.W = VIEWPORT_W/self.scale
        self.H = VIEWPORT_H/self.scale

        # street
        self.n_lanes_from = 2  # global
        self.n_lanes_to = 2  # global
        self.lane_width = 50/self.scale
        self.street_center = self.H//2
        self.upper = self.street_center + self.n_lanes_from*self.lane_width
        self.lower = self.street_center - self.n_lanes_to*self.lane_width

        self.street_polys = []
        self.street_polys.append([(0, self.lower), (self.W, self.lower), (self.W, self.upper), (0, self.upper)])

        # street boarder
        self.boarder_polys = []
        self.boarder_polys.append([(0, self.lower), (self.W, self.lower)])
        self.boarder_polys.append([(0, self.upper), (self.W, self.upper)])

        # lane markings (dashed)
        self.lane_polys = []

        dash_len = self.W/N_DASHES
        space = dash_len * 0.3
        x1 = 0
        for x in np.linspace(dash_len, self.W, N_DASHES):
            x2 = x - space
            # street "to"
            self.lane_polys.append([(x1, self.lower+self.lane_width), (x2, self.lower+self.lane_width)])
            # street "from"
            self.lane_polys.append([(x1, self.upper-self.lane_width), (x2, self.upper-self.lane_width)])
            x1 = x

        # road markings (full)
        self.road_polys = []
        self.road_polys.append([(0, self.street_center), (self.W, self.street_center)])

        # start line
        self.start_line = []
        self.x_start_0 = self.W*0.1
        self.x_start_1 = self.W*0.9
        self.start_line.append([(self.x_start_0, self.lower), (self.x_start_0, self.street_center)])
        self.start_line.append([(self.x_start_1, self.street_center), (self.x_start_1, self.upper)])

        # end line
        self.end_line = []
        self.x_end_0 = self.W*0.9
        self.x_end_1 = self.W*0.1
        self.end_line.append([(self.x_end_0, self.lower), (self.x_end_0, self.street_center)])
        self.end_line.append([(self.x_end_1, self.street_center), (self.x_end_1, self.upper)])

        # gras space
        self.grass_polys = []
        self.grass_polys.append([(0, self.lower), (self.W, self.lower), (self.W, 0), (0, 0)])
        self.grass_polys.append([(0, self.upper), (self.W, self.upper), (self.W, self.H), (0, self.H)])

        # cars
        car_width = dynamic_car_model.SIZE * (dynamic_car_model.WHEEL_W * 2 \
                                         + (dynamic_car_model.WHEELPOS[1][0]-dynamic_car_model.WHEELPOS[1][0]))
        if car_positioning in ["random", "aligned", "opposite"]:
            self.car_init = self.init_car_positions()  # x, y, angle, street_id
        elif car_positioning == "user":
            assert all(v is not None for v in self.car_init)

        for car_id in range(self.num_agents):

            # Create car at location with given angle
            self.cars[car_id] = dynamic_car_model.Car(self.world, self.car_init[car_id][2], self.car_init[car_id][0], self.car_init[car_id][1])
            self.car_colors[car_id] = CAR_COLORS[car_id % len(CAR_COLORS)]
            self.cars[car_id].hull.color = self.car_colors[car_id]

            self.set_car_history(car_id=car_id)

            self.recording_time = [self.t]

            # This will be used to identify the car that touches a particular tile.
            for wheel in self.cars[car_id].wheels:
                wheel.car_id = car_id

        # reset recording parameters
        self.start_recording_x = [self.x_start_0, self.x_start_1]
        self.stop_recording_x = [self.x_end_0, self.x_end_1]
        self.car_street_id = [pos[4] for pos in self.car_init]

        return

    def step(self, action):
        """ Run environment for one timestep.

        Parameters:
            action(np.ndarray): Numpy array of shape (num_agents,3) containing the
                commands for each car. Each command is of the shape (steer, gas, brake).
        """

        if action is not None:
            # NOTE: re-shape action as input action is flattened
            action = np.reshape(action, (self.num_agents, -1))
            for car_id, car in enumerate(self.cars):
                car.steer(-action[car_id][0])
                car.gas(action[car_id][1])
                car.brake(action[car_id][2])

        # log actions
        for car_id in range(self.num_agents):
            self.car_actions[car_id].append(action[car_id].tolist())

        # execute action
        for car in self.cars:
            car.step(1.0/FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self.render()

        # append current car location
        for car_id in range(self.num_agents):
            wheels = []
            for wheel in self.cars[car_id].wheels:
                wheels.append(np.array(wheel.worldCenter))
            car_center = line_intersection((wheels[2], wheels[1]), (wheels[0], wheels[3]))
            car_angle = self.cars[car_id].hull.angle
            self.cars_history[car_id].append(car_center + (car_angle,))

        self.recording_time.append(self.t)

        # start and limit recording

        # start recording - one of both cars passed the start line

        if not self.start_recording:
            for car_id in range(self.num_agents):
                if self.car_street_id[car_id] == 0:
                    if self.cars_history[car_id][-1][0] > self.start_recording_x[self.car_street_id[car_id]]:
                        self.start_recording = True
                else:
                    if self.cars_history[car_id][-1][0] < self.start_recording_x[self.car_street_id[car_id]]:
                        self.start_recording = True
            if self.start_recording:
                self.start_recording_id = len(self.recording_time)
                print("Start recording now, @ n = {} | t = {:6.3f}.".format(len(self.recording_time), self.t))

        # end recording - both cars passed the finish line
        if not all(self.end_recording):
            for car_id in range(self.num_agents):
                if self.car_street_id[car_id] == 0:
                    if self.cars_history[car_id][-1][0] > self.stop_recording_x[self.car_street_id[car_id]]:
                        self.end_recording[car_id] = True
                else:
                    if self.cars_history[car_id][-1][0] < self.stop_recording_x[self.car_street_id[car_id]]:
                        self.end_recording[car_id] = True
            if all(self.end_recording):
                self.end_recording_id = len(self.recording_time)-1
                print("Stop recording now, @ n = {} | t = {}.".format(len(self.recording_time), self.t))
                print("Number of recording time steps: {}".format(self.end_recording_id - self.start_recording_id))
                rec_start_t = self.recording_time[self.start_recording_id]
                rec_stop_t = self.recording_time[self.end_recording_id]
                print("Total recording time: {:6.3f} seconds.".format(rec_stop_t - rec_start_t))

        return self.state

    def render(self):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/self.scale, 0, VIEWPORT_H/self.scale)

        # street
        for p in self.street_polys:
            self.viewer.draw_polygon(p, color=(0.6, 0.6, 0.6))

        # street boarder
        for p in self.boarder_polys:
            self.viewer.draw_polyline(p, color=(0.3, 0.3, 0.3), linewidth=10)

        # lane markings
        for p in self.lane_polys:
            self.viewer.draw_polyline(p, color=(1.0, 1.0, 1.0), linewidth=3)

        # road markings
        for p in self.road_polys:
            self.viewer.draw_polyline(p, color=(1.0, 1.0, 1.0), linewidth=5)

        # start
        for p in self.start_line:
            self.viewer.draw_polyline(p, color=(1.0, 0.31, 0.31), linewidth=5)

        # end
        for p in self.end_line:
            self.viewer.draw_polyline(p, color=(0.4, 0.6, 1.0), linewidth=5)

        # grass
        for p in self.grass_polys:
            self.viewer.draw_polygon(p, color=(0.3, 0.9, 0.3))

        # cars
        for car in self.cars:
            car.draw(self.viewer)

        return self.viewer.render(return_rgb_array=True)

    def get_recorded_data(self):

        timesteps = np.array(self.recording_time)
        traj = np.array([car_traj for car_traj in self.cars_history])
        actions = np.array(self.car_actions)
        # append zero actions at the end of the sequence to make equal number of time steps
        actions = np.append(actions, np.zeros((actions.shape[0], 1, actions.shape[2])), 1)

        data = np.zeros((self.num_agents, len(timesteps), 1 + 3 + 3))
        data[:, :, 0] = timesteps
        data[:, :, 1:4] = traj
        data[:, :, 4:] = actions

        return data

    def get_meta_data(self):

        meta_data = {}
        meta_data["general"] = {"scenario": "highway",
                                "fps": FPS,
                                "num_agents": self.num_agents}
        meta_data["recording"] = {"start_id": self.start_recording_id,
                                  "stop_id": self.end_recording_id}
        meta_data["window"] = {"scale": self.scale,
                               "viewport_w": VIEWPORT_W,
                               "viewport_h": VIEWPORT_H,
                               "road_color": ROAD_COLOR,
                               "n_dashes": N_DASHES,
                               "car_colors": CAR_COLORS,
                               "w": self.W, "h": self.H}
        meta_data["scene"] = {"n_lanes_from": self.n_lanes_from,
                              "n_lanes_to": self.n_lanes_to,
                              "lane_width": self.lane_width,
                              "street_center": self.street_center,
                              "south_boarder": self.lower,
                              "north_boarder": self.upper,
                              "street_polys": self.street_polys,
                              "street_boarder_lines": self.boarder_polys,
                              "lane_markings_dashed": self.lane_polys,
                              "road_markings": self.road_polys,
                              "start_line": self.start_line,
                              "finish_line": self.end_line,
                              "gras_polys": self.grass_polys}
        meta_data["cars"] = []
        for car in self.cars:
            meta_data["cars"].append(car.get_meta())

        for car_id in range(self.num_agents):
            meta_data["cars"][car_id]["init_pos"] = self.car_init[car_id]

        meta_data["annotation"] = {"behavior_types": ["normal", "abnormal"],
                                   "subclass_labels": self.labels,
                                   "acting_agents": self.acting_agents}

        return meta_data

    def set_global_label(self, behavior):
        print("\nLabel pool:")
        for k, v in self.labels[behavior].items():
            print("[{:2}]:\t{}".format(k, v))

        label_id = int(input("Pick a label: "))
        label = self.labels[behavior][label_id]

        label_comment = ""
        if label == "else":
            label_comment = input("Please describe more detailed: ")

        print("\nActing agent:")
        for k, v in self.acting_agents.items():
            print("[{:2}]:\t{}".format(k, v))
        acting_agent = int(input("Pick the main actor: "))

        return label_id, label, label_comment, acting_agent

if __name__=="__main__":

    ####### CONFIG ######
    drivers = {0: "Controller1", 1: "Controller2"}
    car_positioning = "random"  # "aligned", "opposite"

    ####### KEEP UNCHANGED ########

    # export config
    output_root = os.path.join(os.getcwd(), "..", "recordings")
    recording_day = datetime.now().strftime("%Y_%m_%d")
    subdirs = ["normal", "abnormal", "invalid"]
    output_dirs = [os.path.join(output_root, recording_day, subdir) for subdir in subdirs]
    for od in output_dirs:
        if not os.path.isdir(od):
            os.makedirs(od)

    # car control
    from pyglet.window import key
    NUM_CARS = 2  # Supports key control of two cars, but can simulate as many as needed

    # Specify key controls for cars
    CAR_CONTROL_KEYS = [[key.LEFT, key.RIGHT, key.UP, key.DOWN],
                        [key.A, key.D, key.W, key.S]]

    a = np.zeros((NUM_CARS, 3))

    def key_press(k, mod):
        global restart, stopped, CAR_CONTROL_KEYS
        if k == 0xff1b: stopped = True  # Terminate on esc.
        if k == 0xff0d: restart = True  # Restart on Enter.

        # Iterate through cars and assign them control keys (mod num controllers)
        for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
            if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]:  a[i][0] = -1.0
            if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1]: a[i][0] = +1.0
            if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]:    a[i][1] = +1.0
            if k == CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]:  a[i][
                2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        global CAR_CONTROL_KEYS

        # Iterate through cars and assign them control keys (mod num controllers)
        for i in range(min(len(CAR_CONTROL_KEYS), NUM_CARS)):
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][0]  and a[i][0]==-1.0: a[i][0] = 0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][1] and a[i][0]==+1.0: a[i][0] = 0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][2]:    a[i][1] = 0
            if k==CAR_CONTROL_KEYS[i % len(CAR_CONTROL_KEYS)][3]:  a[i][2] = 0

    render = True

    # create environment
    env = MultiCarCountryRoad(NUM_CARS, car_positioning=car_positioning)
    env.seed()
    env.reset()
    env.render()

    # set keyboard functionality
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    # run simulation
    isopen = True
    stopped = False

    while isopen and not stopped:

        env.step(a)

        if render:
            still_open = env.render()

    # Get output directories
    export_path = None
    seq_ids = [len(glob.glob(os.path.join(p, '*.npy')))+1 for p in output_dirs]

    print("\n\n#####################")
    seq_valid = None
    while seq_valid is None:
        usr_in = input("Valid recording [y/n]? ")
        if usr_in == "y":
            seq_valid = True
        elif usr_in == "n":
            usr_in_check = input("Are you sure to move this sequence to trash [y/n]? ")
            if usr_in_check == "y":
                seq_valid = False
                export_path = os.path.join(output_dirs[2], "{:06}".format(seq_ids[2]))
            elif usr_in_check == "n":
                seq_valid = True

    print("\n")
    behavior = "normal"
    while export_path is None:
        scene_type = input("Normal scenario [y/n]? ")
        if scene_type == "y":
            export_path = os.path.join(output_dirs[0], "{:06}".format(seq_ids[0]))
        elif scene_type == "n":
            export_path = os.path.join(output_dirs[1], "{:06}".format(seq_ids[1]))
            behavior = "abnormal"

    # global annotation
    # global sequence labels
    if seq_valid:
        if behavior == "normal":
            label_id, label, label_comment, acting_agent = env.set_global_label(behavior)
        else:
            label_id, label, label_comment, acting_agent = env.set_global_label(behavior)
        annotation = {"class": behavior,
                      "subclass_id": label_id,
                      "subclass": label,
                      "subclass_comment": label_comment,
                      "acting_agent": acting_agent}
    else:
        annotation = {"class": "",
                      "subclass_id": "",
                      "subclass": "",
                      "subclass_comment": "",
                      "acting_agent": ""}

    # meta data
    meta_data = env.get_meta_data()
    meta_data["general"]["drivers"] = drivers
    meta_data["annotation"]["scene"] = annotation

    # trajectory data
    data = env.get_recorded_data()

    # export trajectories and meta data
    np.save(export_path + "_data.npy", data)

    with open(export_path + "_meta.json", "w") as fout:
        json.dump(meta_data, fout)

    print("\nExport sequence to {}.".format(export_path + "[...]"))
    print("\n\n#####################\n")
    print("Number of sequences...")
    seq_ids = [len(glob.glob(os.path.join(p, '*.npy'))) for p in output_dirs]
    sd_str = ""
    for sd in subdirs:
        sd_str += "{:<10}".format(sd)
    n_str = ""
    for n in seq_ids:
        n_str += "{:<10}".format(n)
    print(sd_str)
    print(n_str)

    print("\nDone.")
