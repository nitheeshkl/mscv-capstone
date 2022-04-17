import sys
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams, sigmoid_alpha_blend

from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.transforms.transform3d import Transform3d, Rotate, Translate, Scale
from pytorch3d.structures import join_meshes_as_batch, join_meshes_as_scene

# setup cuda/cpu device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device=device)
    # clear cache
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

# We scale normalize and center the target mesh to fit in a sphere of radius 1
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
# to its original center and scale.  Note that normalizing the target mesh,
# speeds up the optimization but is not necessary!
def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    return mesh


# load mesh obj
def load_meshes(cfg, normalize=True):
    obj_mesh = load_objs_as_meshes([cfg.obj_model], device=device)
    bin_mesh = load_objs_as_meshes([cfg.bin_model], device=device)

    if normalize:
        return normalize_mesh(obj_mesh), normalize_mesh(bin_mesh)
    else:
        return obj_mesh, bin_mesh


# custom Renderer to get depth
class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


# Custome Shader class to obtain object mask
class MaskShader(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        N, H, W, K = fragments.pix_to_face.shape
        device = fragments.pix_to_face.device

        # background mask
        is_background = fragments.pix_to_face[..., 0] < 0
        is_background = is_background.unsqueeze(3)
        bg = torch.zeros((N, H, W, 1)).to(device)
        fg = torch.stack([torch.ones(H, W, 1) * i for i in range(N + 1, 1, -1)]).to(
            device
        )
        images = bg + (~is_background * fg)
        return images  # (N, H, W, 1) RGBA image


def get_depth_renderer(image_size, cameras, lights):

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, lights=lights, cameras=cameras),
    )

    return renderer


def get_mask_renderer(image_size):
    sigma = 1e-4
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=50,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=MaskShader()
    )

    return renderer


def generate_single_image(
    obj_mesh, bin_mesh, depth_renderer, mask_renderer, cameras, cfg
):
    # get tote box bounds
    bin_verts = bin_mesh.verts_list()[0]
    min_xyz, _ = torch.min(bin_verts, dim=0)
    max_xyz, _ = torch.max(bin_verts, dim=0)
    (minx, miny, minz) = min_xyz.cpu().numpy()
    (maxx, maxy, maxz) = max_xyz.cpu().numpy()

    border_padding = 0.5  # to ensure objects are inside tote
    x = torch.tensor(
        np.random.uniform(minx + border_padding, maxx - border_padding, cfg.num_objects)
    )
    y = torch.tensor(
        np.random.uniform(miny + border_padding, maxy - border_padding, cfg.num_objects)
    )
    z = torch.tensor(
        np.random.uniform(minz + border_padding, maxz - border_padding, cfg.num_objects)
    )
    centers = torch.stack([x, y, z], dim=1)
    # random rotation angles for each object.
    angles = torch.deg2rad(torch.randint(0, 360, (cfg.num_objects, 3)))
    # rotation transformation
    rotate = Rotate(euler_angles_to_matrix(angles, "XYZ"), device=device)
    # tranlation transformation
    translate = Translate(centers, device=device)
    # combined transformation for each object
    transform = Transform3d(device=device).compose(rotate).compose(translate)

    meshes = obj_mesh.extend(cfg.num_objects)  # create num_object meshes
    # transform each object mesh
    meshes = meshes.update_padded(transform.transform_points(meshes.verts_padded()))

    rendered_images, depth = depth_renderer(
        join_meshes_as_scene([meshes, bin_mesh]), cameras=cameras
    )
    rgb_img = rendered_images[0, :, :, :3].cpu().numpy()
    depth_img = depth.squeeze().cpu().numpy()

    del rendered_images, depth, depth_renderer
    torch.cuda.empty_cache()

    # generate masks for only obj meshes without bin
    rendered_images = mask_renderer(meshes, cameras=cameras)
    masks = rendered_images.cpu().numpy()
    single_mask = masks.max(axis=0)[:, :, 0]
    fg = masks.sum(axis=0) > 0
    filled_mask = np.ones((fg.shape[0], fg.shape[1], 3)) * 255 * fg

    return rgb_img, depth_img, single_mask, filled_mask


def normalize_depth(img):
    img_norm = (img * 255) / img.max()
    # min max scale according to SD-MaskRCNN depth values
    img_norm = img_norm * ((177 - 97) / (230 - 175))
    img_norm = img_norm - 175 * ((177 - 97) / (230 - 175)) + 97
    return (
        img_norm.astype(np.uint8)[:, :, None]
        * np.ones(3, dtype=np.uint8)[None, None, :]
    )


def generate_data(cfg):

    # load object and bin/tray mesh
    obj_mesh, bin_mesh = load_meshes(cfg)

    # scale and transform bin mesh as per object sizes
    scale = Scale(4, device=device)
    # rotate bin to face camera
    rotate = Rotate(
        euler_angles_to_matrix(torch.deg2rad(torch.tensor([180, 0, 0])), "XYZ"),
        device=device,
    )
    # move bin ahead of camera in z axis
    translate = Translate(0, 0, 6, device=device)
    transform = (
        Transform3d(device=device).compose(scale).compose(rotate).compose(translate)
    )
    bin_mesh_tranformed = bin_mesh.clone()
    bin_mesh_tranformed = bin_mesh_tranformed.update_padded(
        transform.transform_points(bin_mesh_tranformed.verts_padded())
    )

    # setup camera
    cameras = FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0),
        T=torch.tensor([[0, 0, 0]]),  # place it at origin
        device=device,
    )
    # setup lights
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    # setup renderers
    depth_renderer = get_depth_renderer(cfg.image_size, cameras, lights)
    mask_renderer = get_mask_renderer(cfg.image_size)

    for i in tqdm(range(cfg.num_images), desc="generating ", total=cfg.num_images):
        rgb_img, depth_img, modal_segmask, filled_segmask = generate_single_image(
            obj_mesh, bin_mesh_tranformed, depth_renderer, mask_renderer, cameras, cfg
        )

        img_file = f"image_{i:06d}.png"
        rgb_file = os.path.join(cfg.output_dir, "color_ims", img_file)
        rgb_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
        rgb_img.save(rgb_file)
        depth_file = os.path.join(cfg.output_dir, "depth_ims", img_file)
        depth_img = Image.fromarray(normalize_depth(depth_img))
        depth_img.save(depth_file)
        modal_segmask_file = os.path.join(cfg.output_dir, "modal_segmasks", img_file)
        modal_segmask_img = Image.fromarray(modal_segmask.astype(np.uint8))
        modal_segmask_img.save(modal_segmask_file)
        filled_segmask_file = os.path.join(cfg.output_dir, "segmasks_filled", img_file)
        filled_segmask_img = Image.fromarray(filled_segmask.astype(np.uint8))
        filled_segmask_img.save(filled_segmask_file)

        torch.cuda.empty_cache()


# initialize data generation
def initialize(cfg):
    # create output dirs
    if not os.path.exists(cfg.data_generation.output_dir):
        os.makedirs(cfg.data_generation.output_dir)
    # create dir each type of data
    for dir in ["color_ims", "depth_ims", "modal_segmasks", "segmasks_filled"]:
        img_dir = os.path.join(cfg.data_generation.output_dir, dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    initialize(cfg)
    generate_data(cfg["data_generation"])


if __name__ == "__main__":
    main()
