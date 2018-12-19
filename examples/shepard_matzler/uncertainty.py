import argparse
import colorsys
import math
import os
import random
import sys
import time

import cupy as cp
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from chainer.backends import cuda

sys.path.append(".")

import gqn
import rtx
from gqn.preprocessing import make_uint8, preprocess_images
from hyperparams import HyperParameters
from model import Model


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def rotate_query_viewpoint(angle_rad, num_generation, xp):
    view_radius = 3
    eye = (view_radius * math.sin(angle_rad),
           view_radius * math.sin(angle_rad),
           view_radius * math.cos(angle_rad))
    center = (0, 0, 0)
    yaw = gqn.math.yaw(eye, center)
    pitch = gqn.math.pitch(eye, center)
    query_viewpoints = xp.array(
        (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw), math.cos(pitch),
         math.sin(pitch)),
        dtype=np.float32)
    query_viewpoints = xp.broadcast_to(
        query_viewpoints, (num_generation, ) + query_viewpoints.shape)
    return query_viewpoints


def get_available_axis_and_direction(space, pos):
    ret = []
    # x-axis
    for direction in (-1, 1):
        abs_pos = (pos[0] + direction, pos[1], pos[2])
        if space[abs_pos] == True:
            continue
        ret.append((0, direction))
    # y-axis
    for direction in (-1, 1):
        abs_pos = (pos[0], pos[1] + direction, pos[2])
        if space[abs_pos] == True:
            continue
        ret.append((1, direction))
    # z-axis
    for direction in (-1, 1):
        abs_pos = (pos[0], pos[1], pos[2] + direction)
        if space[abs_pos] == True:
            continue
        ret.append((2, direction))

    return ret


def generate_random_block_positions(num_cubes):
    assert num_cubes > 0

    current_relative_pos = (0, 0, 0)
    block_locations = [current_relative_pos]
    block_abs_locations = np.zeros(
        (num_cubes * 2 - 1, num_cubes * 2 - 1, num_cubes * 2 - 1), dtype=bool)
    p = num_cubes - 1
    current_absolute_pos = (p, p, p)
    block_abs_locations[current_absolute_pos] = True

    for _ in range(num_cubes - 1):
        available_axis_and_direction = get_available_axis_and_direction(
            block_abs_locations, current_absolute_pos)
        axis, direction = random.choice(available_axis_and_direction)
        offset = [0, 0, 0]
        offset[axis] = direction
        new_relative_pos = (offset[0] + current_relative_pos[0],
                            offset[1] + current_relative_pos[1],
                            offset[2] + current_relative_pos[2])
        block_locations.append(new_relative_pos)
        current_relative_pos = new_relative_pos
        current_absolute_pos = (
            new_relative_pos[0] + p,
            new_relative_pos[1] + p,
            new_relative_pos[2] + p,
        )
        block_abs_locations[current_absolute_pos] = True

    position_array = []
    center_of_gravity = [0, 0, 0]

    for location in block_locations:
        shift = 1
        position = (shift * location[0], shift * location[1],
                    shift * location[2])

        position_array.append(position)

        center_of_gravity[0] += position[0]
        center_of_gravity[1] += position[1]
        center_of_gravity[2] += position[2]

    center_of_gravity[0] /= num_cubes
    center_of_gravity[1] /= num_cubes
    center_of_gravity[2] /= num_cubes

    return position_array, center_of_gravity


def generate_block_positions(num_cubes):
    position_array = [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 2),
        (0, 0, 1),
        (0, 0, 2),
    ]
    center_of_gravity = [0, 0, 0]

    for position in position_array:
        center_of_gravity[0] += position[0]
        center_of_gravity[1] += position[1]
        center_of_gravity[2] += position[2]

    center_of_gravity[0] /= num_cubes
    center_of_gravity[1] /= num_cubes
    center_of_gravity[2] /= num_cubes

    return position_array, center_of_gravity


def build_scene(color_array):
    # Generate positions of each cube
    cube_position_array, shift = generate_block_positions(args.num_cubes)
    assert len(cube_position_array) == args.num_cubes

    # Place block
    scene = rtx.Scene(ambient_color=(0, 0, 0))
    for position in cube_position_array:
        geometry = rtx.BoxGeometry(1, 1, 1)
        geometry.set_position((
            position[0] - shift[0],
            position[1] - shift[1],
            position[2] - shift[2],
        ))
        material = rtx.LambertMaterial(0.3)
        mapping = rtx.SolidColorMapping(random.choice(color_array))
        cube = rtx.Object(geometry, material, mapping)
        scene.add(cube)

    # Place lights
    size = 50
    group = rtx.ObjectGroup()
    geometry = rtx.PlainGeometry(size, size)
    geometry.set_rotation((0, math.pi / 2, 0))
    geometry.set_position((-10, 0, 0))
    material = rtx.EmissiveMaterial(10, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)

    geometry = rtx.PlainGeometry(size, size)
    geometry.set_rotation((0, -math.pi / 2, 0))
    geometry.set_position((10, 0, 0))
    material = rtx.EmissiveMaterial(1, visible=False)
    mapping = rtx.SolidColorMapping((1, 1, 1))
    light = rtx.Object(geometry, material, mapping)
    group.add(light)

    group.set_rotation((-math.pi / 3, math.pi / 4, 0))
    scene.add(group)

    return scene


def main():
    try:
        os.mkdir(args.figure_directory)
    except:
        pass

    #### Model ####
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_path)
    model = Model(hyperparams, snapshot_directory=args.snapshot_path)
    if using_gpu:
        model.to_gpu()
    print(hyperparams)

    #### Renderer ####
    # Set GPU device
    rtx.set_device(args.gpu_device)

    # Initialize colors
    color_array = []
    for n in range(args.num_colors):
        hue = n / (args.num_colors - 1)
        saturation = 0.9
        lightness = 1
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, lightness)
        color_array.append((red, green, blue, 1))

    screen_width = args.image_size
    screen_height = args.image_size

    # Setting up a raytracer
    rt_args = rtx.RayTracingArguments()
    rt_args.num_rays_per_pixel = 2048
    rt_args.max_bounce = 4
    rt_args.supersampling_enabled = False

    cuda_args = rtx.CUDAKernelLaunchArguments()
    cuda_args.num_threads = 64
    cuda_args.num_rays_per_thread = 32

    renderer = rtx.Renderer()
    render_buffer = np.zeros(
        (screen_height, screen_width, 3), dtype=np.float32)

    camera = rtx.OrthographicCamera()

    #### Figure ####
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("GQN")

    axis_observation = fig.add_subplot(1, 2, 1)
    axis_observation.axis("off")
    axis_observation.set_title("Observation")

    axis_generation = fig.add_subplot(1, 2, 2)
    axis_generation.axis("off")
    axis_generation.set_title("Generation")

    for scene_index in range(100):
        scene = build_scene(color_array)

        eye_scale = 3
        total_frames_per_rotation = 24
        artist_frame_array = []

        observation_viewpoint_angle_rad = 0
        for k in range(5):
            eye = tuple(p * eye_scale for p in [
                math.cos(observation_viewpoint_angle_rad),
                math.sin(observation_viewpoint_angle_rad), 0
            ])
            center = (0, 0, 0)
            camera.look_at(eye, center, up=(0, 1, 0))

            renderer.render(scene, camera, rt_args, cuda_args, render_buffer)

            # Convert to sRGB
            frame = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
            frame = np.uint8(frame * 255)
            frame = cv2.bilateralFilter(frame, 3, 25, 25)

            observation_viewpoint_angle_rad += math.pi / 20

            yaw = gqn.math.yaw(eye, center)
            pitch = gqn.math.pitch(eye, center)
            ovserved_viewpoint = np.array(
                eye + (math.cos(yaw), math.sin(yaw), math.cos(pitch),
                       math.sin(pitch)),
                dtype=np.float32)
            ovserved_viewpoint = ovserved_viewpoint[None, None, ...]

            observed_image = frame.astype(np.float32)
            observed_image = preprocess_images(observed_image, add_noise=False)
            observed_image = observed_image[None, None, ...]
            observed_image = observed_image.transpose((0, 1, 4, 2, 3))

            if using_gpu:
                ovserved_viewpoint = to_gpu(ovserved_viewpoint)
                observed_image = to_gpu(observed_image)

            representation = model.compute_observation_representation(
                observed_image, ovserved_viewpoint)

            query_viewpoint_angle_rad = 0
            for t in range(total_frames_per_rotation):
                artist_array = []

                query_viewpoint = rotate_query_viewpoint(
                    query_viewpoint_angle_rad, 1, xp)
                query_viewpoint = rotate_query_viewpoint(math.pi / 6, 1, xp)
                generated_image = model.generate_image(query_viewpoint,
                                                       representation)
                generated_image = make_uint8(generated_image[0])

                artist_array.append(
                    axis_observation.imshow(
                        frame, interpolation="none", animated=True))
                artist_array.append(
                    axis_generation.imshow(generated_image, animated=True))

                query_viewpoint_angle_rad += 2 * math.pi / total_frames_per_rotation
                artist_frame_array.append(artist_array)
                anim = animation.ArtistAnimation(
                    fig,
                    artist_frame_array,
                    interval=1 / 24,
                    blit=True,
                    repeat_delay=0)
                anim.save(
                    "{}/shepard_matzler_uncertainty_{}.mp4".format(
                        args.figure_directory, scene_index),
                    writer="ffmpeg",
                    fps=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", "-cubes", type=int, default=5)
    parser.add_argument("--num-colors", "-colors", type=int, default=12)
    parser.add_argument(
        "--figure-directory", "-fig", type=str, default="figures")
    args = parser.parse_args()
    main()
