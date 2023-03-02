import os
import math
import torch
import argparse
import skimage.io
import skimage.transform
from torchvision.transforms import ToTensor
# from matplotlib.backends.backend_pdf import PdfPages
from train import build_model
from utils import img_as, pad, crop

# WEIGHTS_PATH = os.path.join('tps_model.pt')
def is_image_file(x):
    ext = os.path.splitext(x)[1].lower()
    return not x.startswith('.') and ext in ['.png', '.jpeg', '.jpg', '.tif', '.tiff']


def get_processors(img_shape, rescale_factor):
    h, w = img_shape[0], img_shape[1],

    def pre_process(image, dapi_only=True):
        image = img_as('float32')(image)
        new_h, new_w = int(math.ceil(h/64) * 64), int(math.ceil(w/64) * 64)
        image = pad(image, pad_width=[(0, new_h-h), (0, new_w-w)])

        if dapi_only:
            image = image[..., 2:]
        image = skimage.transform.rescale(image, scale=rescale_factor, multichannel=True, order=3)

        return image

    def post_process(masks):
        masks = skimage.transform.rescale(masks, scale=1/rescale_factor, multichannel=True, order=3)
        masks[..., 0] = 0.  # ignore bg
        return masks[:h, :w, :]
    
    return pre_process, post_process


def main(args):
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    model = build_model(weights_path=args.model)
    model.eval()
    model.to(device)

    if os.path.isdir(args.input):
        if args.output is None:
            args.output = f"{args.input}_res"
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        image_files = [
            (os.path.join(args.input, x), 
             os.path.join(args.output, f"{os.path.splitext(x)[0]}_res.png"))
            for x in os.listdir(args.input) if is_image_file(x)
        ]
    else:
        if args.output is None:
            root, ext = os.path.splitext(args.input)
            args.output = f"{root}_res.png"
        image_files = [(args.input, args.output)]

    for input_file, output_file in image_files:
        print(f"Analyze slides: {input_file}")
        img = skimage.io.imread(input_file)
        pre_processor, post_processor = get_processors(img.shape, args.rescale_factor)

        img = pre_processor(img, dapi_only=True)
        with torch.no_grad():
            inputs = ToTensor()(img)[None].to(device)
            outputs = model(inputs)
            masks = outputs[0].permute(1, 2, 0).detach().cpu()
        res = post_processor(masks.numpy())

        skimage.io.imsave(output_file, (255*res).astype('uint8'))
    if args.export:
        root, ext = os.path.splitext(args.model)
        script_file = f"{root}.torchscript.pt"
        print(f"Export torchscript model: {script_file}")
        torch.jit.save(torch.jit.script(model), script_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TPS inference.', add_help=False)
    parser.add_argument('--input', required=True, type=str, help="Input image file.")
    parser.add_argument('--model', default='./pretrained/tps_model.pt', type=str, help="Model file or weights path." )
    parser.add_argument('--output', default=None, type=str, help="Output file name. default use `input_mask.ext`.")
    parser.add_argument('--rescale_factor', default=1.0, type=float, help="Rescale factor for inference.")
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'], help='Run inference on GPU or CPU.')
    parser.add_argument('--export', action='store_true', help='Whether to export torch script model.')

    args = parser.parse_args()
    # Example
    # args = parser.parse_args(
    #     [
    #         '--input', 
    #         '/home2/s190548/work_personal/software/GoZDeep/data/2w_Control_Gls2-0244-F456-L1.tif',
    #         '--device', 'cpu',
    #         '--rescale_factor', '1', 
    #         '--output', '/home2/s190548/work_personal/software/GoZDeep/data/2w_Control_Gls2-0244-F456-L1_mask.tif'
    #         ]
    #     )
    main(args)
