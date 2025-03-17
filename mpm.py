import taichi as ti
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from LSystem import *
import imageio

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E * 2
la = E
max_steps = 20000
steps = 1024
gravity = 3.8

act_strength = 5

particleCeiling = 3000

def allocate_fields():
    ti.root.dense(ti.i, n_actuators).place(amplitude)
    ti.root.dense(ti.i, n_actuators).place(frequency)
    ti.root.dense(ti.i, n_actuators).place(phase)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = new_F.determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])
        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)
        act_id = actuator_id[p]
        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), ti.f32) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass

bound = 3
coeff = 0.5

@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out = ti.Vector([0.0, 0.0])
        if i > n_grid - bound and v_out[0] > 0:
            v_out = ti.Vector([0.0, 0.0])
        if j < bound and v_out[1] < 0:
            v_out = ti.Vector([0.0, 0.0])
        if j > n_grid - bound and v_out[1] > 0:
            v_out = ti.Vector([0.0, 0.0])
        grid_v_out[i, j] = v_out

@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * new_v
        C[f + 1, p] = new_C

@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = amplitude[i] * ti.sin(frequency[i] * t * dt + phase[i])
        actuation[t, i] = ti.tanh(act)

@ti.kernel
def compute_x_avg(s: ti.i32) -> ti.f32:
    total: ti.f32 = 0.0
    for i in range(n_particles):
        total += x[s, i][0]
    return total / n_particles

@ti.kernel
def compute_x_min(s: ti.i32) -> ti.f32:
    min_x: ti.f32 = 1e9  # a large number to start with
    for i in range(n_particles):
        min_x = ti.min(min_x, x[s, i][0])
    return min_x

# A Taichi function that computes the average rotation for time step s
@ti.func
def compute_avg_rotation_frame(s: ti.i32) -> ti.f32:
    # Compute the center of mass (COM) at time s-1.
    com = ti.Vector([0.0, 0.0])
    for p in range(n_particles):
        com += x[s - 1, p]
    com /= n_particles

    total_angle: ti.f32 = 0.0
    for p in range(n_particles):
        # Compute displacement vectors from the COM for s-1 and s.
        disp_prev = x[s - 1, p] - com
        disp_curr = x[s, p] - com

        # Calculate the 2D cross product (a scalar) and dot product.
        cross = disp_prev[0] * disp_curr[1] - disp_prev[1] * disp_curr[0]
        dot = disp_prev[0] * disp_curr[0] + disp_prev[1] * disp_curr[1]

        # Compute the signed rotation using atan2.
        angle = ti.atan2(cross, dot)
        total_angle += angle
    return total_angle / n_particles

# A kernel that computes the cumulative sum of average rotations.
@ti.kernel
def compute_rotation_cumsum(total_steps: ti.i32):
    # We set the cumulative rotation at frame 0 to zero.
    rotation_cumsum[0] = 0.0
    # For each subsequent frame, compute the average rotation and add it to the previous sum.
    for s in range(1, total_steps):
        rotation_cumsum[s] = compute_avg_rotation_frame(s)

def compute_x_min_avgd(s: int) -> float:
    # Extract the x coordinates (only the x component) for simulation step s
    x_np = x.to_numpy()[s][:, 0]
    # Determine the number of particles corresponding to the lowest 10%
    num_bottom = max(1, int(0.1 * n_particles))
    # Sort the x coordinates in ascending order
    sorted_x = np.sort(x_np)
    # Compute and return the average of the lowest 10% of x values
    return float(np.mean(sorted_x[:num_bottom]))

def compute_avg_rotation(s: int) -> float:
    if s < 1:
        return 0
    # Get the positions for time steps s-1 and s
    pos_prev = x.to_numpy()[s-1]
    pos_curr = x.to_numpy()[s]
    
    # Compute center of mass at time step s-1 (all particles have equal mass)
    com = np.mean(pos_prev, axis=0)
    
    # Compute displacement vectors from the COM for each particle
    disp_prev = pos_prev - com  # at time s-1
    disp_curr = pos_curr - com  # at time s
    
    # For 2D vectors, the signed angle between disp_prev and disp_curr is given by:
    # angle = arctan2(cross, dot), where
    # cross = x_prev * y_curr - y_prev * x_curr, and
    # dot   = x_prev * x_curr + y_prev * y_curr
    cross = disp_prev[:, 0] * disp_curr[:, 1] - disp_prev[:, 1] * disp_curr[:, 0]
    dot = disp_prev[:, 0] * disp_curr[:, 0] + disp_prev[:, 1] * disp_curr[:, 1]
    
    # Compute signed angles for each particle
    angles = np.arctan2(cross, dot)
    
    # Return the average signed rotation over all particles
    return float(np.mean(angles))


# The function 'advance' now just performs one simulation step.
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

# The forward function runs the simulation for a fixed number of steps.
def forward(total_steps=steps):
    for s in range(total_steps - 1):
        advance(s)

# -------------------------------------------------------------------
# Geometry and scene definition 
# -------------------------------------------------------------------
class Rectangle:
    def __init__(self, origin, angle, length, width):
        self.origin = np.array(origin)
        self.angle = angle
        self.length = length
        self.width = width
        self.actuatorID = 0
        self.minParticleIndex = 0
        self.maxParticleIndex = 0
    
    def getVertices(self):
        centerline = self.length * self.getNormals()[0]
        centerlineNormal = self.width/2 * self.getNormals()[1]
        return [
            self.origin + centerlineNormal,
            self.origin + centerline + centerlineNormal,
            self.origin + centerline - centerlineNormal,
            self.origin - centerlineNormal,
        ]
    
    def getNormals(self):
        return [
            np.array([math.cos(self.angle), math.sin(self.angle)]),
            np.array([math.sin(self.angle), -math.cos(self.angle)])
        ]
    
    def intersects(self, other):
        normals = self.getNormals() + other.getNormals()
        for normal in normals:
            thisProjection = [np.dot(vertex, normal) for vertex in self.getVertices()]
            otherProjection = [np.dot(vertex, normal) for vertex in other.getVertices()]
            if max(thisProjection) < min(otherProjection) or min(thisProjection) > max(otherProjection):
                return False
        return True
    
    def contains(self, point):
        normals = self.getNormals()
        for normal in normals:
            thisProjection = [np.dot(vertex, normal) for vertex in self.getVertices()]
            otherProjection = np.dot(point, normal)
            if max(thisProjection) < otherProjection or min(thisProjection) > otherProjection:
                return False
        return True

class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.amplitude = []
        self.frequency = []
        self.phase = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

        self.structure_width = 0
        self.structure_height = 0

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

    # L-system parser
    def build_robot_skeleton(self, axiom, lengthMultiplier, densityMultiplier=3):
        maxActuatorID = -1
        newActuatorID = -1
        drawn = []
        stack = []
        globalMinX = 0
        globalMinY = 0
        globalMaxX = 0
        globalMaxY = 0 
        currentState = [np.array([0, 0]), 0]
        stack.append(currentState.copy())
        #print(f"axiom is {axiomString(axiom)}")
        #printAxiom(axiom)
        commandCount = 0
        def process_commands(commands):
            nonlocal commandCount
            commandCount+=1
            if commandCount > 100:
                return
            #print(f"called on {axiomString(commands)}")
            nonlocal currentState, stack, drawn, maxActuatorID, newActuatorID, globalMinX, globalMinY, globalMaxX, globalMaxY
            i = 0
            while i < len(commands):
                if self.n_particles > particleCeiling:
                    return
                #print(f"{self.n_particles} particles in scene currently")
                command = commands[i]
                #print(f"On {command.type.name} with params {command.params}")

                # If we encounter a repeat block, collect everything until the matching '}'
                if command.type.name == '{':
                    nested = 1
                    i += 1  # Skip the opening brace
                    block_start = i
                    while i < len(commands) and nested > 0:
                        if commands[i].type.name == '{':
                            nested += 1
                        elif commands[i].type.name == '}':
                            nested -= 1
                        i += 1
                    # Extract the block (exclude the closing '}')
                    repeat_block = commands[block_start:i-1]
                    repeatTimes = min(int(abs(commands[i-1].params[0])), 100)
                    for _ in range(repeatTimes):
                        process_commands(repeat_block)
                    continue  # Continue with the next command (i already points after the '}')

                # Process other commands.
                elif command.type.name == '[':
                    stack.append(currentState.copy())
                elif command.type.name == ']':
                    if len(stack) != 0:
                        currentState = stack.pop()
                # Allow for command objects (with a type attribute) or raw string symbols:
                elif command.type.name == "+":
                    currentState[1] += command.params[0]
                elif command.type.name == "-":
                    currentState[1] -= command.params[0]
                elif command.type.name == "F" or command.type.name == "G":
                    newSegment = Rectangle(currentState[0], currentState[1], abs(command.params[0])*lengthMultiplier, abs(command.params[1])*lengthMultiplier)
                    if command.type.name == "F":
                        newActuatorID = maxActuatorID + 1
                        self.amplitude.append(np.clip(command.params[2], -10, 10))
                        self.frequency.append(np.clip(command.params[3], -10, 10))
                        self.phase.append(command.params[4])
                    elif command.type.name == "G":
                        newActuatorID = -1
            
                    lengthParticleCount = int(newSegment.length / dx) * densityMultiplier
                    widthParticleCount = int(newSegment.width / dx) * densityMultiplier

                    if lengthParticleCount*widthParticleCount > particleCeiling:
                        print(f"WARN: single segment required more than {particleCeiling} particles")
                        return

                    if widthParticleCount > 0 and lengthParticleCount > 0:
                        lengthStep = newSegment.length / lengthParticleCount
                        widthStep = newSegment.width / widthParticleCount

                        newSegment.minParticleIndex = len(self.x)
                        for i_length in range(lengthParticleCount):
                            for j in range(widthParticleCount):
                                particleCoordinates = (lengthStep * i_length * newSegment.getNormals()[0] +
                                                    (widthStep * j - newSegment.width/2) * newSegment.getNormals()[1] +
                                                    currentState[0])
                                self.x.append([float(particleCoordinates[0]), float(particleCoordinates[1])])
                                globalMinX = min(particleCoordinates[0], globalMinX)
                                globalMinY = min(particleCoordinates[1], globalMinY)
                                globalMaxX = max(particleCoordinates[0], globalMaxX)
                                globalMaxY = max(particleCoordinates[1], globalMaxY)
                                self.actuator_id.append(newActuatorID)
                                self.particle_type.append(1)
                                self.n_particles += 1
                                self.n_solid_particles += 1

                        newSegment.maxParticleIndex = len(self.x) - 1
                        if maxActuatorID < newActuatorID:
                            maxActuatorID = newActuatorID
                        newSegment.actuatorID = newActuatorID
                        drawn.append(newSegment)
                        currentState[0] = currentState[0] + newSegment.getNormals()[0] * newSegment.length

                    else: print(f"WARN: tried to create a segment with no particles; dimensions were {command.params[0]}x{command.params[1]}")
                i += 1

        # Start processing from the full axiom
        process_commands(axiom)
        self.set_n_actuators(maxActuatorID + 1)
        for point in self.x:
            point[0] += (-globalMinX + self.offset_x)
            point[1] += (-globalMinY + self.offset_y)
        self.structure_width = globalMaxX
        self.structure_height = globalMaxY

def visualize(s, folder, actuator_id_copy, gui, save=False, filename=None):
    aid = np.array(actuator_id_copy)
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    currentPos = compute_x_min_avgd(s)
    gui.text(f"position: {currentPos:.3f}", pos=(0.05, 0.95), font_size=12, color=0x0)
    gui.text(f"distance traveled: {currentPos-startingPos:.3f}", pos=(0.05, 0.93), font_size=12, color=0x0)
    gui.text(f"total rotation: {totals[s]:.3f} deg", pos=(0.05, 0.91), font_size=12, color=0x0)
    if save:
        os.makedirs(folder, exist_ok=True)
        if not filename:
            gui.show(f'{folder}/{s:04d}.png')
        else:
            gui.show(f'{folder}/{filename}.png')
    else:
        gui.show()

def scoreStructure(structure, folder, filename, vis=False, render=False, renderFrameLimit = 0, crop=False):
    os.makedirs(folder, exist_ok=True)

    ti.reset()

    ti.f32 = ti.f32
    ti.init(default_fp=ti.f32, arch=ti.gpu, flatten_if=True)

    global actuator_id, particle_type, x, v, grid_v_in, grid_m_in, grid_v_out, C, F
    global amplitude, frequency, phase, actuation
    global rotation_cumsum

    # Define Taichi field types
    scalar = lambda: ti.field(dtype=ti.f32)
    vec = lambda: ti.Vector.field(dim, dtype=ti.f32)
    mat = lambda: ti.Matrix.field(dim, dim, dtype=ti.f32)

    # Fields for simulation
    actuator_id = ti.field(ti.i32)
    particle_type = ti.field(ti.i32)

    x, v = vec(), vec()
    grid_v_in, grid_m_in = vec(), scalar()
    grid_v_out = vec()
    C, F = mat(), mat()

    amplitude = scalar()
    frequency = scalar()
    phase = scalar()
    actuation = scalar()

    steps=5000

    # Initialize the scene and build the robot
    scene = Scene()
    scene.set_offset(0.05, 0.05)
    
    scene.build_robot_skeleton(structure, 0.01, 4)

    scene.finalize()
    if n_actuators == 0:
        print("WARN: structure was unactuated")
        return -1000000
    if scene.n_particles > particleCeiling: # prevent structures with lots of particles from crashing my computer
        print(f"WARN: exceeded particle ceiling; {scene.n_particles}")
    allocate_fields()
    rotation_cumsum = ti.field(dtype=ti.f32, shape=steps)

    # Set the actuation parameters 
    for i in range(n_actuators):
        amplitude[i] = scene.amplitude[i]
        frequency[i] = scene.frequency[i]
        phase[i] = scene.phase[i]

    # Initialize particle positions and other properties
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    # Run the simulation for a fixed number of steps with fixed weights
    forward(steps)

    compute_rotation_cumsum(steps)
    rotations = rotation_cumsum.to_numpy()
    rotations = rotations * 180/np.pi
    global totals
    totals = []
    totals.append(0)
    for s in range(1, steps):
        totals.append(totals[s-1]+rotations[s])
    
    global startingPos
    startingPos = compute_x_min_avgd(0)
    score = compute_x_min_avgd(steps-1)-startingPos
    #score -= 0.01*abs(totals[-1])
    
    actuator_id_copy = scene.actuator_id.copy()
    #save a single screenshot of the structure
    offscreen_res = (1280, 1280)
    offscreen_gui = ti.GUI("Screenshot Rendering", res=offscreen_res, show_gui=False, background_color=0xFFFFFF)
    visualize(20, f"{folder}/screenshots", actuator_id_copy, offscreen_gui, save=True, filename=filename)    
    img = imageio.imread(f"{folder}/screenshots/{filename}.png")
    height, width = img.shape[:2]
    cropped = img[height-600:height-30, :600]
    imageio.imwrite(f"{folder}/screenshots/{filename}.png", cropped)

    if vis:
        gui = ti.GUI("MPM Simulation", (640, 640), background_color=0xFFFFFF)

        started = False
        while not started:
            
            # Draw a message on the window so the user sees it
            gui.text("space to start", (0.35, 0.5), font_size=24, color=0x0)
            # Process all events
            for e in gui.get_events():
                if e.type == ti.GUI.PRESS and e.key == ' ':
                    started = True
            gui.show()  # Refresh the window
        
        # Visualize selected simulation frames
        for s in range(15, steps, 16):
            visualize(s, 'mpm_simulation/', actuator_id_copy, gui)

    if render:
        tempFolder = "tempImages"
        timeControl = 16 # was 8
        renderFrameLimit = min(renderFrameLimit, steps)
        offscreen_res = (640, 640)
        offscreen_gui = ti.GUI("High-Res Rendering", res=offscreen_res, show_gui=False, background_color=0xFFFFFF)
        for s in range(0, renderFrameLimit, timeControl):
            visualize(s, tempFolder, actuator_id_copy, offscreen_gui, save=True)
            print(f"Rendered {s}/{renderFrameLimit}")
            img = imageio.imread(f"{tempFolder}/{s:04d}.png")
            height, width = img.shape[:2]

            if crop:
                # Compute the structure center in simulation space and convert to pixel space.
                # Here we assume the simulation [0,1] maps to the full GUI resolution.
                particles = x.to_numpy()[s]
                center_sim = np.mean(particles, axis=0)
                center_px = (int(center_sim[0] * width), int((1 - center_sim[1]) * height))

                crop_width = 208
                crop_height = 208

                x1 = int(center_px[0] - crop_width // 2)
                y1 = int(center_px[1] - crop_height // 2)

                # Adjust the coordinates to ensure the crop window fits within the image bounds
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x1 + crop_width > width:
                    x1 = width - crop_width
                if y1 + crop_height > height:
                    y1 = height - crop_height

                # Crop the image using the fixed window size
                cropped = img[y1:y1+crop_height, x1:x1+crop_width]
                imageio.imwrite(f"{tempFolder}/{s:04d}.png", cropped)

        os.makedirs(f"{folder}/renders", exist_ok=True)
        video_filename = f"{folder}/renders/{filename}.mp4"
        frames = [imageio.imread(f"{tempFolder}/{s:04d}.png") for s in range(0, renderFrameLimit, timeControl)]
        with imageio.get_writer(video_filename, fps=60, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)

    return score



  
    
 