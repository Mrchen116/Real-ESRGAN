import argparse
import cv2
import glob
import os

from realesrgan import RealESRGANer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
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

    upsampler = RealESRGANer(
        scale=args.netscale,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    path = args.input
    videoname, extension = os.path.splitext(os.path.basename(path))
    print('Testing', videoname)

    cap = cv2.VideoCapture(path)
    writer = None
    save_path = os.path.join(args.output, f'{videoname}_{args.suffix}.mp4')

    while True:
        ret, img = cap.read()
        if not ret:
            break
        h, w = img.shape[0:2]
        if writer is None:
            if args.outscale == 4:
                new_h, new_w = h * 4, w * 4
            elif args.outscale == 2:
                new_h, new_w = h * 2, w * 2
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), 25.0, (new_w, new_h))
            if max(h, w) > 1000 and args.netscale == 4:
                import warnings
                warnings.warn('The input image is large, try X2 model for better performace.')
            if max(h, w) < 500 and args.netscale == 2:
                import warnings
                warnings.warn('The input image is small, try X4 model for better performace.')

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except Exception as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            writer.write(output)


if __name__ == '__main__':
    main()
