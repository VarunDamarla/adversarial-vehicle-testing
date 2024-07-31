import csv
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transform
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)

DIR_HERE = Path(__file__).resolve().parent

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
cwd = str(Path.cwd())
if "/src/adversarial_vehicle_testing/" not in cwd:
    cwd += "/src/adversarial_vehicle_testing/"
model_path = cwd + "DAVE2v3.pt"
model = torch.load(model_path, map_location=torch.device("cuda")).eval()

i = -20.0
end = 25.0
x_step = 0.001
rows = []
while i < end + x_step:
    x_change = torch.tensor(float(i), requires_grad=True)
    x_change.retain_grad()
    y_change = torch.tensor(float(0), requires_grad=True)
    y_change.retain_grad()
    z_change = torch.tensor(float(0), requires_grad=True)
    z_change.retain_grad()

    mesh = load_objs_as_meshes([cwd + "nimrud.obj"], device=device)
    r, t = look_at_view_transform(-20, 170, 50)
    t[0][0] += x_change
    t[0][1] += y_change
    t[0][2] += z_change
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

    device = fragments.pix_to_face.device
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
    image_path = cwd + "straight_scene.jpg"
    background = torch.tensor(cv2.imread(image_path), device=device).float() / 255.0
    background.requires_grad = True
    background.retain_grad()
    background_comp = background.unsqueeze(0) * is_background.unsqueeze(-1)
    is_background_comp = torch.flip(images[...,:3], [-1]) * torch.logical_not(is_background.unsqueeze(-1))
    combined = background_comp + is_background_comp
    combined[0].retain_grad()

    pert_image = combined[0].permute(2, 0, 1).unsqueeze(0)
    pert_image.retain_grad()
    transformation = transform.Resize((135, 240))
    resized_pert = transformation(pert_image)
    pert_angle = model(resized_pert)
    pert_angle.retain_grad()

    orig_image = background.permute(2, 0, 1).unsqueeze(0)
    orig_image.retain_grad()
    resized_orig = transformation(orig_image)
    orig_angle = model(resized_orig)
    orig_angle.retain_grad()

    loss = -1 * abs(pert_angle - orig_angle)
    loss.backward(retain_graph=True)
    rows.append([x_change.grad.item(), x_change.item(), pert_angle.item()])
    print(rows[-1][0], rows[-1][1], rows[-1][2])
    i += x_step

with Path(cwd + "x_translation_data.csv").open("w", newline="") as file:
    file.truncate()
    writer = csv.writer(file)
    field = ["Gradient", "Change in X", "Perturbation Angle"]
    writer.writerow(field)
    writer.writerows(rows)
