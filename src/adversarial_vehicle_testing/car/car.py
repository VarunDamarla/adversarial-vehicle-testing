from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transform
from DAVE2pytorch import DAVE2v3
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
cwd = str(Path.cwd())
model_path = cwd + "/DAVE2v3.pt"
model = torch.load(model_path, map_location=torch.device("cuda")).eval()

def render_car_translation_pert(x_change: float, y_change: float, z_change: float, device: torch.device) -> str:
    x_change = torch.tensor(float(x_change), requires_grad=True)
    x_change.retain_grad()
    y_change = torch.tensor(float(y_change), requires_grad=True)
    y_change.retain_grad()
    z_change = torch.tensor(float(z_change), requires_grad=True)
    z_change.retain_grad()

    mesh = load_objs_as_meshes(["nimrud.obj"], device=device)
    r, t = look_at_view_transform(-20, 170, 50)
    t[0][0] += x_change.item()
    t[0][1] += y_change.item()
    t[0][2] += z_change.item()
    t.requires_grad = True
    t.retain_grad()
    cameras = FoVPerspectiveCameras(device=device, R=r, T=t)
    raster_settings = RasterizationSettings(image_size=(225, 400), blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=device, location=[[5.0 , 0.0, 0.0]])
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)

    verts_shape = mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    mesh = mesh.offset_verts(deform_verts)

    fragments = rasterizer.forward(mesh)
    images = shader(fragments, mesh)

    plt.figure(figsize=(400/25, 225/25))
    plt.imshow(images[0, ..., :3].detach().cpu().numpy())
    plt.axis("off")

    device = fragments.pix_to_face.device
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
    image_path = cwd + "/straight_scene.jpg"
    background = torch.tensor(cv2.imread(image_path), device=device).float() / 255.0
    background_comp = background.unsqueeze(0) * is_background.unsqueeze(-1)
    is_background_comp = torch.flip(images[...,:3], [-1]) * torch.logical_not(is_background.unsqueeze(-1))
    combined = background_comp + is_background_comp
    plt.figure(figsize=(400, 225))
    plt.imshow(torch.flip(combined[0, ..., :3], [-1]).detach().cpu().numpy())
    plt.axis("off")
    file_name = f"combined x{x_change.item()}_y{y_change.item()}_z{z_change.item()}.png"
    cv2.imwrite(file_name, combined[0].detach().cpu().numpy() * 255)
    plt.close()
    return file_name, x_change, y_change, z_change, t

def generate_steer_tensor(file_name: str, device: torch.device, model: DAVE2v3) -> torch.Tensor:
    pert_image: torch.Tensor
    processed_pert: torch.Tensor
    resized_pert: torch.Tensor
    pert_angle: torch.Tensor
    pert_image = torch.tensor(cv2.imread(file_name), device=device)
    pert_image = pert_image.detach().cpu()
    processed_pert = model.process_image(pert_image).to(torch.device("cuda"))
    transformation = transform.Resize((135, 240))
    resized_pert = transformation(processed_pert)
    pert_angle = model(resized_pert)
    return pert_angle

def generate_mlt_translations(n: int, c_x: int, c_y: int, c_z: int, d: torch.device, m:DAVE2v3) -> "list[torch.Tensor]":
    tensors = []
    for i in range(n):
        x = c_x * i
        y = c_y * i
        z = c_z * i
        file_name, x_change, y_change, z_change, t = render_car_translation_pert(x, y, z, d)
        x_change.retain_grad()
        y_change.retain_grad()
        z_change.retain_grad()
        name = f"{cwd}/{file_name}"
        tensor = generate_steer_tensor(name, d, m)
        result = [tensor, x_change, y_change, z_change, t]
        tensors.append(result)
    return tensors

def generate_original_steer() -> torch.Tensor:
    return generate_steer_tensor(cwd + "/straight_scene.jpg", device, model)

def generate_steer_gradient(device: torch.device, model: DAVE2v3, num: int, x:int, y:int, z:int) -> None:
    orig_angle: torch.Tensor
    orig_angle = generate_original_steer()
    data = generate_mlt_translations(num, x, y, z, device, model)
    model.zero_grad()
    losses = []
    orig_angle.retain_grad()
    for p in range(num):
        data[p][0].retain_grad()
        data[p][1].retain_grad()
        data[p][2].retain_grad()
        data[p][3].retain_grad()
        losses.append(-1 * abs(orig_angle - data[p][0]))
        print(f"LossP{p}: {losses[-1]}")
        losses[-1].backward(retain_graph=True)
        print(f"GradientP{p}: {data[p][1].grad}")

# Understanding how the translation works
def main() -> None:
    generate_steer_gradient(device, model, 10, 1, 0, 0)

main()
