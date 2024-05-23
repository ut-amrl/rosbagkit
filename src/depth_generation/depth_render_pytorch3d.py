import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.utils import ico_sphere
from tqdm import tqdm


import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    look_at_view_transform,
    MeshRasterizer,
    PerspectiveCameras,
)
from pytorch3d.ops import points_to_volumes
from pytorch3d.renderer.cameras import PerspectiveCameras
import matplotlib.pyplot as plt

device = torch.device("cpu")

# Define the camera parameters
cameras = PerspectiveCameras(
    R=torch.eye(3).unsqueeze(0), T=torch.zeros(1, 3), K=torch.eye(4).unsqueeze(0)
)
cameras = PerspectiveCameras(
    image_size=[img_size],
    # R=R[None],
    # T=T[None],
    focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
    principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
    in_ndc=False,
)


raster_settings = RasterizationSettings(
    image_size=(1080, 1440),
    blur_radius=0.0,
    faces_per_pixel=1,
    # max_faces_per_bin=20000
)

rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


# Function to create a sphere at a given center
def create_sphere(center, radius=0.1, level=2):
    sphere_mesh = ico_sphere(level=level)
    verts = sphere_mesh.verts_list()[0] * radius + center
    faces = sphere_mesh.faces_list()[0]
    return verts, faces


# Assuming you have 3D points in a tensor of shape (N, 3)
points = torch.rand((10, 3))  # Replace with your 3D points

# Generate spheres for each point
verts_list = []
faces_list = []
for i, point in tqdm(enumerate(points)):
    verts, faces = create_sphere(point)
    faces += i * verts.shape[0]  # Update face indices
    verts_list.append(verts)
    faces_list.append(faces)

# Combine all spheres into a single mesh
all_verts = torch.cat(verts_list, dim=0)
all_faces = torch.cat(faces_list, dim=0)

# Create the final mesh
# mesh = Meshes(verts=[all_verts], faces=[all_faces])


# convert points to voxel grid and make a mesh
voxel_size = 0.05
verts, faces, aux = points_to_volumes(points, voxel_size=voxel_size)
mesh = Meshes(verts=[verts], faces=[faces])


print(mesh.verts_packed().shape)

fragments = rasterizer(mesh)

# visualize the rasterized points
plt.figure(figsize=(10, 10))
plt.imshow(fragments.pix_to_face.squeeze().cpu().numpy())
plt.grid("off")
plt.axis("off")

plt.show()


class HardDepthShader(ShaderBase):
    """
    Renders the Z distances of the closest face for each pixel. If no face is
    found it returns the zfar value of the camera.
    Output from this shader is [N, H, W, 1] since it's only depth.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = HardDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = self.cameras

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar
        return zbuf
