import trimesh
import pyrender
import time

# get center
mesh = pyrender.Mesh.from_trimesh(trimesh.load("data/surface00000.stl"))
center = mesh.centroid

nodes = list()
iterations = 1000
for i in range(iterations):
    mesh_path = "data/surface{:0>5d}.stl".format(i)
    print("Reading in surface data: %s" % mesh_path)
    mesh = trimesh.load(mesh_path)
    mesh.apply_transform([
        [1,0,0,-center[0]],
        [0,1,0,-center[1]],
        [0,0,1,-center[2]],
        [0,0,0,1]
    ])
    dm = pyrender.Mesh.from_trimesh(mesh)
    node = pyrender.Node(
        scale=[0.0125,0.0125,0.0125],
        mesh=dm)
    nodes.append(node)

scene = pyrender.Scene(
    bg_color=[0,0,0,0]
)
v = pyrender.Viewer(scene,
    viewport_size=(1280,720),
    use_direct_lighting=True,
    all_wireframe=True,
    run_in_thread=True,
    caption=[{
            "location": 3,
            "text": "",
            "font_name": "OpenSans-Regular",
            "font_pt": 40,
            "color": [200,200,200,255],
            "scale": 1.0,
            "align": 0
        }],
    record=False#,
#    rotate=True,
#    rotate_rate=0.25,
#    rotate_axis=[0,1,0]
    )

for i in range(iterations):
    it = "Iteration %d" % i
    print(it)
    v.render_lock.acquire()
    v.viewer_flags['caption'][0]["text"] = it
    if i > 0:
        scene.remove_node(nodes[i-1])
    scene.add_node(nodes[i])
    v.render_lock.release()
    time.sleep(0.033333333333333)
#v.close_external()
#v.save_gif("test.gif")
