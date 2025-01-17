import argparse
import cv2
import glob
import os
from tqdm import tqdm

from realesrgan import RealESRGANer

class Inference:
    # Namespace(alpha_upsampler='realesrgan', ext='auto', face_enhance=False, half=False, 
    # input='/root/first-order-model/log/ori1_crop_pic.mp4', 
    # model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth', netscale=4, 
    # output='/root/first-order-model/log/enhance_ori1_crop_pic.mp4', 
    # outscale=4, pre_pad=0, suffix='out', tile=800, tile_pad=10)
    def __init__(self, face_enhance=False, half=False, 
                 model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth', netscale=4, outscale=4, 
                 pre_pad=0, tile=800, tile_pad=10):
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half)

        if face_enhance:
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
                upscale=args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)
        
        self.outscale = outscale
        self.face_enhance = face_enhance
    
    def inference(self, img):
        try:
            if self.face_enhance:
                _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = self.upsampler.enhance(img, outscale=self.outscale)
        except Exception as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            return None
        else:
            return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results.mp4', help='Output video name')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=800, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()
    print("_________+------------+______________args :", args)
    
    # os.makedirs(args.output, exist_ok=True)

    path = args.input
    videoname, extension = os.path.splitext(os.path.basename(path))
    print('Testing', videoname)

    cap = cv2.VideoCapture(path)
    fps = cap.get(5)
    frame_cnt = int(cap.get(7))
    writer = None
    # save_path = os.path.join(args.output, f'{videoname}_{args.suffix}.mp4')
    save_path = args.output
    infer = Inference(args.face_enhance, args.half,
                     args.model_path, args.netscale, args.outscale, args.pre_pad, args.tile, args.tile_pad)
    for i in tqdm(range(frame_cnt)):
        ret, img = cap.read()
        if not ret:
            break
        elif writer is None:
            h, w = img.shape[0:2]
            if args.outscale == 4:
                new_h, new_w = h * 4, w * 4
            elif args.outscale == 2:
                new_h, new_w = h * 2, w * 2
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (new_w, new_h))
            if max(h, w) > 1000 and args.netscale == 4:
                import warnings
                warnings.warn('The input image is large, try X2 model for better performace.')
            if max(h, w) < 500 and args.netscale == 2:
                import warnings
                warnings.warn('The input image is small, try X4 model for better performace.')

        output = infer.inference(img)
        writer.write(output)


if __name__ == '__main__':
    main()
