#!/usr/bin/env python3

import os
import argparse
import trimesh
import pyrender
import time
import zipfile


def render(surface_path: str, video_path: str = None, loop: bool = False):
    """
    Create rendering of surface deformation
    """
    # open surfaces file
    surface_dir = os.path.dirname(surface_path)
    surfaces = zipfile.ZipFile(surface_path, "r")

    # get surface list
    surface_list = surfaces.namelist()
    iterations = len(surface_list)

    # get center of mesh (use first surface file)
    surfaces.extract(surface_list[0], path=surface_dir)
    mesh = pyrender.Mesh.from_trimesh(trimesh.load(os.path.join(surface_dir, surface_list[0])))
    center = mesh.centroid
    os.remove(os.path.join(surface_dir, surface_list[0]))

    # read in surfaces
    nodes = list()
    for mesh_path in surface_list:
        surfaces.extract(mesh_path, path=surface_dir)
        print("Reading in surface data: %s" % mesh_path, end="\r")
        mesh = trimesh.load(os.path.join(surface_dir, mesh_path))
        os.remove(os.path.join(surface_dir, mesh_path))
        mesh.apply_transform([[1, 0, 0, -center[0]], [0, 1, 0, -center[1]], [0, 0, 1, -center[2]], [0, 0, 0, 1]])
        dm = pyrender.Mesh.from_trimesh(mesh)
        node = pyrender.Node(scale=[0.01, 0.01, 0.01], mesh=dm)
        nodes.append(node)
    print("")

    # create scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    v = pyrender.Viewer(
        scene,
        viewport_size=(1280, 720),
        use_direct_lighting=True,
        all_wireframe=True,
        run_in_thread=True,
        caption=[
            {
                "location": 3,
                "text": "",
                "font_name": "OpenSans-Regular",
                "font_pt": 40,
                "color": [200, 200, 200, 255],
                "scale": 1.0,
                "align": 0,
            }
        ],
        record=bool(video_path),
        rotate=True,
        rotate_rate=0.25,
        rotate_axis=[0, 1, 0],
    )

    # display surfaces frame by frame
    iteration_limit = iterations
    if loop:
        iterations *= int(3600 / (iterations * 0.033333333333333))
    try:
        for i in range(iterations):
            c = i - (i // iteration_limit) * iteration_limit
            it = "Iteration %d" % c
            print(it, end="\r")
            v.render_lock.acquire()
            v.viewer_flags["caption"][0]["text"] = it
            if c > 0:
                scene.remove_node(nodes[c - 1])
            elif i > 0 and i % iteration_limit == 0:
                scene.remove_node(nodes[iteration_limit - 1])
            scene.add_node(nodes[c])
            v.render_lock.release()
            time.sleep(0.033333333333333)
        v.close_external()
    except KeyboardInterrupt:
        pass
    print("")

    # save video
    if video_path:
        dirpath = os.path.dirname(video_path)
        os.makedirs(dirpath, exist_ok=True)
        print("Saving video to file...")
        v.save_gif(os.path.join(dirpath, "temp.gif"))
        os.system("ffmpeg -i {} {}".format(os.path.join(dirpath, "temp.gif"), video_path))
        os.remove(os.path.join(dirpath, "temp.gif"))
        print("{} successfully saved.".format(os.path.basename(video_path)))


def main():
    # create command line parser
    parser = argparse.ArgumentParser(
        description="Renders surface deformation evolution",
        epilog="Author: Andrew Van, vanandrew@wustl.edu, 12/15/2020",
    )
    parser.add_argument("surfaces", help="Surfaces to render")
    parser.add_argument("-s", "--save_mp4", help="Saves an mp4 output")
    parser.add_argument("-l", "--loop", action="store_true", help="Loop the render (1 hour)")

    # parse arguments
    args = parser.parse_args()

    # call render function
    render(
        surface_path=os.path.abspath(args.surfaces),
        video_path=os.path.abspath(args.save_mp4) if args.save_mp4 else None,
        loop=args.loop,
    )
