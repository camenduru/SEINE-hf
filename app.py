import gradio as gr
from image_to_video import model_i2v_fun, get_input, auto_inpainting, setup_seed
from omegaconf import OmegaConf
import torch
from diffusers.utils.import_utils import is_xformers_available
import torchvision 
from utils import mask_generation_before
import os 
import cv2

config_path = "/mnt/petrelfs/zhouyan/project/i2v/configs/sample_i2v.yaml"
args = OmegaConf.load(config_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------- get model ---------------
# model_i2V = model_i2v_fun()
# model_i2V.to("cuda")

# vae, model, text_encoder, diffusion = model_i2v_fun(args)
# vae.to(device)
# model.to(device)
# text_encoder.to(device)

# if args.use_fp16:
#     vae.to(dtype=torch.float16)
#     model.to(dtype=torch.float16)
#     text_encoder.to(dtype=torch.float16)

# if args.enable_xformers_memory_efficient_attention and device=="cuda":
#     if is_xformers_available():
#         model.enable_xformers_memory_efficient_attention()
#     else:
#         raise ValueError("xformers is not available. Make sure it is installed correctly")


css = """
h1 {
  text-align: center;
}
#component-0 {
  max-width: 730px;
  margin: auto;
}
"""

def infer(prompt, image_inp, seed_inp, ddim_steps):
    setup_seed(seed_inp)
    args.num_sampling_steps = ddim_steps
    ###先测试Image的返回类型
    print(prompt, seed_inp, ddim_steps, type(image_inp))
    img = cv2.imread(image_inp)
    new_size = [img.shape[0],img.shape[1]]
    # if(img.shape[0]==512 and img.shape[1]==512):
    #     args.image_size = [512,512]
    # elif(img.shape[0]==320 and img.shape[1]==512):
    #     args.image_size = [320, 512]
    # elif(img.shape[0]==292 and img.shape[1]==512):
    #     args.image_size = [292,512]
    # else:
    #     raise ValueError("Please enter image of right size")
    # print(args.image_size)
    args.image_size = new_size

    vae, model, text_encoder, diffusion = model_i2v_fun(args)
    vae.to(device)
    model.to(device)
    text_encoder.to(device)

    if args.use_fp16:
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)

    if args.enable_xformers_memory_efficient_attention and device=="cuda":
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    video_input, reserve_frames = get_input(image_inp, args)
    video_input = video_input.to(device).unsqueeze(0)
    mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device)
    masked_video = video_input * (mask == 0)
    prompt = "tilt up, high quality, stable "
    prompt = prompt + args.additional_prompt
    video_clip = auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,)
    video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
    torchvision.io.write_video(os.path.join(args.save_img_path,  prompt+ '.mp4'), video_, fps=8)


    
    # video = model_i2V(prompt, image_inp, seed_inp, ddim_steps)
    
    return os.path.join(args.save_img_path,  prompt+ '.mp4')



def clean():
    # return gr.Image.update(value=None, visible=False), gr.Video.update(value=None)
    return gr.Video.update(value=None)


title = """
    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            SEINE: Image-to-Video generation
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Apply SEINE to generate a video 
        </p>
    </div>
"""



with gr.Blocks(css='style.css') as demo:
    gr.Markdown("<font color=red size=10><center>SEINE: Image-to-Video generation</center></font>")
    with gr.Column(elem_id="col-container"):
        # gr.HTML(title)
        
        with gr.Row():
            with gr.Column():
                image_inp = gr.Image(type='filepath')
                
            with gr.Column():
                
                prompt = gr.Textbox(label="Prompt", placeholder="enter prompt", show_label=True, elem_id="prompt-in")
                
                with gr.Row():
                    # control_task = gr.Dropdown(label="Task", choices=["Text-2-video", "Image-2-video"], value="Text-2-video", multiselect=False, elem_id="controltask-in")
                    ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=250, step=1)
                    seed_inp = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=250, elem_id="seed-in")
                
                # ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=250, step=1)
                
               
                
                submit_btn = gr.Button("Generate video")
                clean_btn = gr.Button("Clean video")

        video_out = gr.Video(label="Video result", elem_id="video-output", width = 800)
        inputs = [prompt,image_inp, seed_inp, ddim_steps]
        outputs = [video_out]
        ex = gr.Examples(
            examples = [["/mnt/petrelfs/zhouyan/project/i2v/The_picture_shows_the_beauty_of_the_sea_.jpg","A video of the beauty of the sea",123,50],
                        ["/mnt/petrelfs/zhouyan/project/i2v/The_picture_shows_the_beauty_of_the_sea.png","A video of the beauty of the sea",123,50],
                        ["/mnt/petrelfs/zhouyan/project/i2v/Close-up_essence_is_poured_from_bottleKodak_Vision.png","A video of close-up essence is poured from bottleKodak Vision",123,50]],
            fn = infer,
            inputs = [image_inp, prompt, seed_inp, ddim_steps],
            outputs=[video_out],
            cache_examples=False


        )
        ex.dataset.headers = [""]
        # gr.Markdown("<center>some examples</center>")
        # with gr.Row():
        #     gr.Image(value="/mnt/petrelfs/zhouyan/project/i2v/The_picture_shows_the_beauty_of_the_sea_.jpg")
        #     gr.Image(value="/mnt/petrelfs/zhouyan/project/i2v/The_picture_shows_the_beauty_of_the_sea.png")
        #     gr.Image(value="/mnt/petrelfs/zhouyan/project/i2v/Close-up_essence_is_poured_from_bottleKodak_Vision.png")
        # with gr.Row():
        #     gr.Video(value="/mnt/petrelfs/zhouyan/project/i2v/The-picture-shows-the-beauty-of-the-sea-and-at-the-sam_slow-motion_0000_11301.mp4")
        #     gr.Video(value="/mnt/petrelfs/zhouyan/project/i2v/The-picture-shows-the-beauty-of-the-sea-and-at-the-sam_slow-motion_0000_6600.mp4")
        #     gr.Video(value="/mnt/petrelfs/zhouyan/project/i2v/Close-up-essence-is-poured-from-bottleKodak-Vision3-50_slow-motion_0000_001.mp4")
    # control_task.change(change_task_options, inputs=[control_task], outputs=[canny_opt, hough_opt, normal_opt], queue=False)
    clean_btn.click(clean, inputs=[], outputs=[video_out], queue=False)
    submit_btn.click(infer, inputs, outputs)
    # share_button.click(None, [], [], _js=share_js)


demo.queue(max_size=12).launch(server_name="0.0.0.0",server_port=7861)


