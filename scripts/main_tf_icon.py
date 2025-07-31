import argparse, os
import PIL
import torch
import re
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import cv2
import time
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import pickle
from typing import Union, Tuple, Optional
from torchvision import transforms as tvt


def load_bg(
    bg_img, 
) :
    image = bg_img
    w, h = image.size        
    print(f"loaded input image of size ({w}, {h}) from {bg_img}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    w = h = 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def resize_with_padding(img, size=512, fill_color=0):
    # 保持比例缩放
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    # 创建新图像并居中粘贴
    if img.mode == 'L' or img.mode == '1':
        new_img = Image.new(img.mode, (size, size), fill_color)
    else:
        new_img = Image.new(img.mode, (size, size), (fill_color,)*3)
    left = (size - img.width) // 2
    top = (size - img.height) // 2
    new_img.paste(img, (left, top))
    print(f"img_size = {img.size}")
    return new_img, left, top, img.width, img.height

def load_fg(
    fg_img, 
    seg_img,
    target_size: tuple,
    box_position: list  # [x1, y1, x2, y2]，像素坐标
):
    batch_size = 1
    # 1. 居中填充到 target_size
    fg_padded, left, top, fg_w, fg_h = resize_with_padding(fg_img, size=target_size[0])
    seg_padded, _, _, _, _ = resize_with_padding(seg_img, size=target_size[0])

    # 2. 转为 tensor
    # a. 前景图 padded -> [-1, 1] Tensor
    ref_image_tensor = 2. * tvt.ToTensor()(fg_padded).unsqueeze(0) - 1.
    ref_image_tensor = ref_image_tensor.repeat(batch_size, 1, 1, 1)
    # b. 分割蒙版 padded -> [0, 1] Tensor (512x512)
    seg_padded_np = np.array(seg_padded).astype(np.float32) / 255.0
    seg_tensor_512 = torch.from_numpy(seg_padded_np).unsqueeze(0).unsqueeze(0)
    seg_tensor_512 = seg_tensor_512.repeat(batch_size, 1, 1, 1)
    # c. 下采样到 latent space (64x64)
    seg_tensor_64 = torch.nn.functional.interpolate(seg_tensor_512, scale_factor=1/8, mode='nearest')

    # 3. 计算前景在 padded 图中的像素坐标 [top, bottom, left, right]
    ref_bbox_px = [top, top + fg_h, left, left + fg_w]

    # 4. 计算目标位置在 latent space 中的坐标 (top, bottom, left, right)
    scale_x = 64 / target_size[0]
    scale_y = 64 / target_size[1]
    latent_bbox = (
        int(box_position[1] * scale_y), # top (y1)
        int(box_position[3] * scale_y), # bottom (y2)
        int(box_position[0] * scale_x), # left (x1)
        int(box_position[2] * scale_x)  # right (x2)
    )
    
    return ref_image_tensor, seg_tensor_512, seg_tensor_64, ref_bbox_px, latent_bbox, fg_w, fg_h


def load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=False):
           
    if inv:
        inv_emb = model.get_learned_conditioning(prompts, inv)
        c = uc = inv_emb
    else:
        inv_emb = None
        
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)
        
    return c, uc, inv_emb
    
    
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of a doggy, ultra realistic",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--ref-img",
        type=list,
        nargs="?",
        help="path to the input image"
    )
    
    parser.add_argument(
        "--seg",
        type=str,
        nargs="?",
        help="path to the input image"
    )
        
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--dpm_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    
    parser.add_argument(
        "--f",
        type=int,
        default=16,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/gys/TFICON/TF-ICON/ckpt/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="the seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument(
        "--root",
        type=str,
        help="",
        default='./inputs/cross_domain'
    ) 
    
    parser.add_argument(
        "--domain",
        type=str,
        help="",
        default='cross'
    ) 
    
    parser.add_argument(
        "--dpm_order",
        type=int,
        help="",
        choices=[1, 2, 3],
        default=2
    ) 
    
    parser.add_argument(
        "--tau_a",
        type=float,
        help="",
        default=0.4
    )
      
    parser.add_argument(
        "--tau_b",
        type=float,
        help="",
        default=0.8
    )
          
    parser.add_argument(
        "--gpu",
        type=str,
        help="",
        default='cuda:0'
    ) 
    parser.add_argument(
        "--position",
        type=tuple,
        help="矩形框位置[[x1, y1, x2, y2]]",
        default='cuda:0'
    ) 
    
    opt = parser.parse_args()       
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # The scale used in the paper
    if opt.domain == 'cross':
        opt.scale = 5.0
        file_name = "cross_domain"
    elif opt.domain == 'same':
        opt.scale = 2.5
        file_name = "same_domain"
    else:
        raise ValueError("Invalid domain")
        
    batch_size = opt.n_samples
    sample_path = os.path.join(outpath, file_name)

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu)    
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    
    for subdir, _, files in os.walk(opt.root):
        for file in files:
            torch.cuda.empty_cache()
            file_path = os.path.join(subdir, file)

            opt.prompt = os.path.basename(subdir).replace("_", " ")
            if file.startswith('background'):
                opt.init_img = file_path
            elif file.startswith('foreground') and not (file.endswith('_centered.png')) :
                opt.ref_img = file_path
            elif file.startswith('location'):
                opt.mask = file_path
            elif file.startswith('segmentation'):
                opt.seg = file_path
            elif file.startswith("position"):
                with open(file_path, 'rb') as positions:
                    opt.position = pickle.load(positions)   
            if file == files[-1]:
                seed_everything(opt.seed)
                
                # 直接从 pickle 文件加载精确的矩形框位置 [x1, y1, x2, y2]
                print(f"Loaded position: {opt.position}")
                x1, y1, x2, y2 = opt.position
                new_w, new_h = x2 - x1, y2 - y1
                
                # 计算中心点百分比
                center_x = x1 + new_w / 2
                center_y = y1 + new_h / 2
                
                center_row_from_top = center_y / 512
                center_col_from_left = center_x / 512 

                prompt = opt.prompt
                data = [batch_size * [prompt]]
                
                # --- 使用新的、清晰的加载和计算流程 ---
                
                # 1. 加载 PIL 对象
                bg_pil = Image.open(opt.init_img).convert("RGB")
                fg_pil = Image.open(opt.ref_img).convert("RGB")
                seg_pil = Image.open(opt.seg).convert("L")
                
                print("fg的大小草尼玛TF-ICON",fg_pil.size,"fg的大小草尼玛TF-ICON",opt.ref_img)
                
                # 2. 加载背景图
                init_image = load_bg(bg_pil).to(device)
                init_image = repeat(init_image.to(device), '1 ... -> b ...', b=batch_size)

                # 3. 加载前景图并获取所有处理好的数据
                ref_image, seg_512, seg_64, ref_bbox_px, latent_bbox, width, height = load_fg(
                    fg_pil, seg_pil, (512, 512), opt.position
                )
                
                print(f"width, height = {width},{height}")
                
                ref_image = ref_image.to(device)
                seg_512 = seg_512.to(device)
                seg_64 = seg_64.to(device)
                # 4. 准备用于不同阶段的蒙版
                
                segmentation_map_save = seg_512.repeat(1, 3, 1, 1) # 用于像素合成
                segmentation_map = seg_64.repeat(1, 4, 1, 1)       # 用于 latent 合成

                # a. padded 图中的 foreground 坐标 [top, bottom, left, right]
                top_rr,bottom_rr, left_rr, right_rr = ref_bbox_px
                height = bottom_rr - top_rr
                width = right_rr-left_rr
                
                # b. 最终粘贴到背景图上的坐标
                target_height, target_width = 512, 512
                center_row_rm = int(center_row_from_top * target_height)
                center_col_rm = int(center_col_from_left * target_width)
                step_h2, rem_h = divmod(height, 2); step_h1 = step_h2 + rem_h
                step_w2, rem_w = divmod(width, 2); step_w1 = step_w2 + rem_w
                
                paste_y_slice = slice(center_row_rm - step_h1, center_row_rm + step_h2)
                paste_x_slice = slice(center_col_rm - step_w1, center_col_rm + step_w2)

                # c. latent space 的目标粘贴坐标

                # --- 像素空间合成 ---
                save_image = init_image.clone()
                
                bg_slice = save_image[..., paste_y_slice, paste_x_slice]
                ref_slice = ref_image[..., top_rr:bottom_rr, left_rr:right_rr]
                seg_slice = segmentation_map_save[..., top_rr:bottom_rr, left_rr:right_rr]

                save_image[..., paste_y_slice, paste_x_slice] = bg_slice * (1 - seg_slice) + ref_slice * seg_slice
                # save the mask and the pixel space composited image
                save_mask = torch.zeros_like(init_image) 
                save_mask[:, :, center_row_rm - step_h1:center_row_rm + step_h2, center_col_rm - step_w1:center_col_rm + step_w2] = 1

                image = Image.fromarray(((save_mask) * 255)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
                image.save('./outputs/mask_bg_fg.jpg')
                save_image[..., paste_y_slice, paste_x_slice] = bg_slice * (1 - seg_slice) + ref_slice * seg_slice

                save_image = torch.clamp(save_image, -1.0, 1.0)

                # 2. 标准公式转换
                image_to_save_numpy = ((save_image[0] + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                # 3. 保存
                image_pil = Image.fromarray(image_to_save_numpy)
                image_pil.save('./outputs/cp_bg_fg.jpg')                 

                precision_scope = autocast if opt.precision == "autocast" else nullcontext
                
                # image composition
                with torch.no_grad():
                    with precision_scope("cuda"):
                        for prompts in data:
                            print(prompts)
                            c, uc, inv_emb = load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=True)
                            
                            if opt.domain == 'same': # same domain
                                init_image = save_image
                            
                            T1 = time.time()
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  
                            ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_image))
                        
                            # 1. 直接使用 load_fg 计算好的 latent_bbox 作为 param
                            #    latent_bbox 的格式是 (top, bottom, left, right)
                            ref_padded_location_latened = [int(ref_bbox_px[0]/8),int(ref_bbox_px[1]/8),int(ref_bbox_px[2]/8),int(ref_bbox_px[3]/8)]
                            
                            print(f"ref_padded_location_latened={ref_padded_location_latened}")
                            # 2. 我们还需要 ref_image 中前景的 latent 坐标，以便从中提取特征

                            ref_top_l =ref_padded_location_latened[0]
                            ref_bottom_l = ref_padded_location_latened[1]
                            ref_left_l = ref_padded_location_latened[2]
                            ref_right_l = ref_padded_location_latened[3]
                            print(f"ref_top_l,ref_bottom_l,ref_left_l,ref_right_={ref_top_l}, {ref_bottom_l}, {ref_left_l}, {ref_right_l}")
                            
                            new_height = ref_padded_location_latened[1] - ref_padded_location_latened[0]
                            new_width = ref_padded_location_latened[3] - ref_padded_location_latened[2]
                            
                            step_height2, remainder = divmod(new_height, 2)
                            step_height1 = step_height2 + remainder
                            step_width2, remainder = divmod(new_width, 2)
                            step_width1 = step_width2 + remainder
                            
                            center_row_rm = int(center_row_from_top * init_latent.shape[2])
                            center_col_rm = int(center_col_from_left * init_latent.shape[3])

                            print (f"center_row_rm={center_row_rm},center_col_rm ={center_col_rm }")
                                          
                            ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_image))
                        
                            shape = [init_latent.shape[1], init_latent.shape[2], init_latent.shape[3]]
                            print(f"init_latent.shape={init_latent.shape},width={width},heigth={height}")
                            z_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                    inv_emb=inv_emb,
                                                    unconditional_conditioning=uc,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    eta=opt.ddim_eta,
                                                    order=opt.dpm_order,
                                                    x_T=init_latent,
                                                    width=width,
                                                    height=height,
                                                    DPMencode=True,
                                                    )
                            
                            z_ref_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                        inv_emb=inv_emb,
                                                        unconditional_conditioning=uc,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        eta=opt.ddim_eta,
                                                        order=opt.dpm_order,
                                                        x_T=ref_latent,
                                                        DPMencode=True,
                                                        width=width,
                                                        height=height,
                                                        ref=True,
                                                        )
                            
                            samples_orig = z_enc.clone()
                        

                            top_rr,bottom_rr, left_rr, right_rr = int(ref_bbox_px[0]/8),int(ref_bbox_px[1]/8),int(ref_bbox_px[2]/8),int(ref_bbox_px[3]/8)
                            param = [latent_bbox[0], latent_bbox[1], latent_bbox[2], latent_bbox[3]]
                            print(f"param before XOR{param}")
                            # inpainting in XOR region of M_seg and M_mask
                            z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                                = z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                                * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                + torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) \
                                * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])

                            samples_for_cross = samples_orig.clone()
                            samples_ref = z_ref_enc.clone()
                            samples = z_enc.clone()

                            # noise composition
                            if opt.domain == 'cross': 
                                samples[:, :, param[0]:param[1], param[2]:param[3]] = torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) 
                                # apply the segmentation mask on the noise
                                samples[:, :, param[0]:param[1], param[2]:param[3]] \
                                        = samples[:, :, param[0]:param[1], param[2]:param[3]].clone() \
                                        * (1 - segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]) \
                                        + z_ref_enc[:, :, top_rr: bottom_rr, left_rr: right_rr].clone() \
                                        * segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]
                            
                            mask = torch.zeros_like(z_enc, device=device)
                            mask[:, :, param[0]:param[1], param[2]:param[3]] = 1
                                                
                            samples, _ = sampler.sample(steps=opt.dpm_steps,
                                                        inv_emb=inv_emb,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        order=opt.dpm_order,
                                                        x_T=[samples_orig, samples.clone(), samples_for_cross, samples_ref, samples, init_latent],
                                                        width=width,
                                                        height=height,
                                                        segmentation_map=segmentation_map,
                                                        param=param,
                                                        mask=mask,
                                                        target_height=target_height, 
                                                        target_width=target_width,
                                                        center_row_rm=center_row_from_top,
                                                        center_col_rm=center_col_from_left,
                                                        tau_a=opt.tau_a,
                                                        tau_b=opt.tau_b,
                                                        )
                                
                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            
                            T2 = time.time()
                            print('Running Time: %s s' % ((T2 - T1)))
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                MyOutputPath=os.path.join(outpath,prompts[0])
                                os.makedirs(MyOutputPath,exist_ok=True)
                                img.save(os.path.join(MyOutputPath, "result.png"))

                del x_samples, samples, z_enc, z_ref_enc, samples_orig, samples_for_cross, samples_ref, mask, x_sample, img, c, uc, inv_emb
                del param, segmentation_map, top_rr, bottom_rr, left_rr, right_rr, target_height, target_width, center_row_rm, center_col_rm
                del init_image, init_latent, save_image, ref_image, ref_latent, prompt, prompts, data



if __name__ == "__main__":
    main()

