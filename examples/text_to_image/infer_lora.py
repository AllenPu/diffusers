from diffusers import StableDiffusionPipeline
import torch
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--description", help = "what you wanna see", type = str, default="naked full body")
args = parser.parse_args()



model_path = "sd-naruto-model-lora"
pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", torch_dtype=torch.float16, safety_checker=None)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")



des = args.description
prompt = "A woman with {}".format(des)
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("test1.png")