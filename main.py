import torch
import torch.optim as optim
import numpy as np
from data import data
from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
import sys
import os
import threading
import json
import random

from scripts.time_log import time_log_module as tlm
from scripts.prompt_enhancer import enhance_prompt
import scripts.pc_stats as ps
from scripts.logger import logger
from scripts.local_api import API


# args : --train = to train the model
#        --overwrite = if added, the script will regenerate the enhanced trainning prompts
#        --server = To activate the local API
#        --generate = To generate a GenMoji

# TODO : Code the local API   | Priority : 2
#        train the model      | Priority : 1
#        Github repo          | Priority : 4
#        UIX (13b)            | Priority : 3
#        iOS/Android version  | Priority : 5 


train = "--train" in sys.argv or "-t" in sys.argv
overwrite = "--overwrite" in sys.argv or "-o" in sys.argv
local_api = "--server" in sys.argv or "-s" in sys.argv
gen = "--generate" in sys.argv or "-g" in sys.argv

discord_webhook = "" # Just leave it empty if you don't want your logs to be send to discord
base_model_path = r"I:\sd1.5\diffusion_pytorch_model.safetensors" # <!> Change it by YOUR Stable Diffusion 1.5 model path <!>
version = 2.0
new_model_name = "genmoji"
new_model_path = f"model/{new_model_name}.safetensors"

if __name__ == "__main__":
    print(f"{tlm()} Start of program.")
    logger = logger(discord_webhook)
    logger.log(f"GenMoji V{str(version)}")
    
    # Check stable diffusion base model path
    if not os.path.exists(base_model_path):
        logger.log(f"Error: Stable Diffusion 1.5 model not found at path: {base_model_path}, please change it by YOUR own path in main.py, check README.md", v=False)
        raise FileNotFoundError(f"{tlm()} Error: Stable Diffusion 1.5 model not found at path: {base_model_path}, please change it by YOUR own path in main.py, check README.md")

    # Load data
    dataset = data()
    logger.log(f"Loading data...")
    dataset.load_data()
    logger.log(f"Enhancing trainning data prompts...")
    stop_event = threading.Event()
    t = threading.Thread(target=ps.monitor, args=(stop_event,))
    t.start()
    enhanced_status = dataset.save_enhanced_prompt(overwrite=overwrite)
    stop_event.set()
    t.join()
    try:
        dataset.load_enhanced_data()
    except:
        logger.log(f"Error :\nEnhancing status: {str(enhaneced_status)}\n{Exception}", v=False)
        raise OSError(f"{tlm()} Error :\nEnhancing status: {str(enhaneced_status)}\n{Exception}")

    logger.log(f"Data loaded. Number of samples: {len(dataset.data)}")

    # Fine tune Stable Diffusion 1.5
    if train: # bruh j'aurais du créer une classe, fait par ChatGPT car TensorFlow >>>
        logger.log(f"Loading Stable Diffusion 1.5 model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_single_file(
            base_model_path, 
            torch_dtype=torch.float16
        ).to("cuda")

        logger.log(f"Configuring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
        )
        
        # Apply LoRA to the UNet
        model = get_peft_model(pipe.unet, lora_config)
        model.print_trainable_parameters()

        # Training loop
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        num_epochs: int = 10
        batch_size: int = 4

        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        logger.log(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(0, num_epochs):
            logger.log(f"Epoch {epoch + 1}/{num_epochs}")
            for i in range(0, len(dataset.data), batch_size):
                batch = dataset.data[i:i + batch_size]
                images = [item[0] for item in batch]
                prompts = [item[1] for item in batch]

                # Convert images to tensors and normalize
                images_tensor = []
                for idx, img in enumerate(images):
                    # Handle different image formats
                    if img.ndim == 2:  # grayscale (H, W)
                        img = np.stack([img, img, img], axis=-1)
                    elif img.ndim == 3:
                        if img.shape[2] == 1:  # grayscale (H, W, 1)
                            img = np.repeat(img, 3, axis=2)
                        elif img.shape[2] == 2:  # grayscale + alpha or malformed
                            # Take first channel and replicate to RGB
                            img = np.stack([img[:, :, 0], img[:, :, 0], img[:, :, 0]], axis=-1)
                            logger.log(f"Warning: Image {idx} in batch had 2 channels, converted to grayscale RGB")
                        elif img.shape[2] == 4:  # RGBA -> RGB
                            img = img[:, :, :3]
                        elif img.shape[2] != 3:
                            logger.log(f"Image {idx} has unexpected shape: {img.shape}", v=False)
                            raise ValueError(f"Image {idx} has unexpected shape: {img.shape}")
                    else:
                        logger.log(f"Image {idx} has unexpected number of dimensions: {img.ndim}", v=False)
                        raise ValueError(f"Image {idx} has unexpected number of dimensions: {img.ndim}")
                
                    # Normalize to [0, 1] if not already
                    if img.max() > 1.0:
                        img = img / 255.0
                    
                    images_tensor.append(torch.from_numpy(img).float().permute(2, 0, 1))
                
                images = torch.stack(images_tensor).to("cuda").half()

                # Forward pass
                # images -> latents
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

                # prompts -> embeddings
                inputs = pipe.tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=pipe.tokenizer.model_max_length
                ).input_ids.to("cuda")
                text_embeds = pipe.text_encoder(inputs)[0]

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (latents.size(0),), 
                    device="cuda"
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Forward through UNet with LoRA
                noise_pred = model(noisy_latents, timesteps, text_embeds).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i // batch_size) % 10 == 0:
                    logger.log(f"Batch {i // batch_size}, Loss: {loss.item()}")

        logger.log(f"Training completed. Saving model...")
        model.save_pretrained(f"model/{new_model_name}")
        pipe.save_pretrained(f"model/{new_model_name}_pipeline")
        logger.log(f"Model saved.")
    else:
        # Load fine-tuned model
        pipe = StableDiffusionPipeline.from_single_file(
            base_model_path, 
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, "model/genmoji")
        pipe.to("cuda")
    
    # Usage test
    if train: # en sois le seul moment ou on voudra test comme ça c après le train
        logger.log(f"Generating an emoji with the fine-tuned model, prompt : 'Flying pig emoji'")
        prompt = "Flying pig emoji"
        enhanced_prompt = enhance_prompt(prompt)
        image = pipe(enhanced_prompt).images[0]
        image.save("test_output.png")
        logger.log(f"Image saved to 'test_output.png'.")
    if gen:
        if "--generate" in sys.argv: # Si ya la commande gen, recup le prompt
            index = sys.argv.index("--generate")
            if index + 1 < len(sys.argv):
                texte = sys.argv[index + 1]
        elif "-g" in sys.argv:
            index = sys.argv.index("-g")
            if index + 1 < len(sys.argv):
                texte = sys.argv[index + 1]
        if texte == None:
            logger.log(f"No prompt was given after the --generate or -g argument, no genmoji will be generated.")
        else:
            gen_ID = random.randint(1, 1000000000000)
            logger.log(f"Generating an emoji with the fine-tuned model, Generation ID : {str(gen_ID)}, prompt : '{texte}'")
            enhanced_prompt = enhance_prompt(enhance_prompt(texte))
            logger.log(f"Enhanced prompt : {enhanced_prompt}")
            steps = 25
            image = pipe(enhanced_prompt, num_inference_steps=steps).images[0]
            image.save(f"generated-{str(gen_ID)}.png")
            logger.log(f"Image saved to 'generated-{str(gen_ID)}.png'.")
    if local_api:
        # lancer une fonction qui vas tourner a l'infinis avec un port ouvert (qui se ferme avec un localhost:*port*/sleep)
        api = API(pipe)
        api.serve()

    logger.log(f"End of program.")
