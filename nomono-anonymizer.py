#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nomono-anonymizer - Face anonymizer on movie scenes

This program anonymizes with the Henohenomoheji (no-mo-no) letters by using the Keypoint R-CNN architecture.

  * https://github.com/foota/nomono-anonymizer

Usage:
  * $ python nomono-anonymizer.py input-video-file output-video-file

References:
  * https://github.com/kkroening/ffmpeg-python
  * https://pytorch.org/docs/stable/torchvision/models.html#keypoint-r-cnn

© 2020 foota
"""

import subprocess
import argparse
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision

import ffmpeg

parser = argparse.ArgumentParser(description="Face anonymizer on movie scenes")
parser.add_argument("infile", help="Input video filename")
parser.add_argument("outfile", help="Output video filename")
parser.add_argument(
    "-d",
    "--device",
    default="cuda",
    action="store",
    help="Device: [cuda | cpu | auto] (default: auto)",
)


def process_nomono(model, frame):
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    frame = torch.tensor(
        frame / 255.0, dtype=torch.float32, device=DEVICE
    )  # host -> device
    frame = torch.transpose(frame, 1, 2)  # (H, C, W)
    frame = torch.transpose(frame, 0, 1)  # (C, H, W)
    frame = torch.unsqueeze(frame, 0)  # (1, C, H, W)
    preds = model(frame)
    cnt = 0
    for pred in preds:
        for box, label, score, keypoints, keypoints_scores in zip(
            pred["boxes"],
            pred["labels"],
            pred["scores"],
            pred["keypoints"],
            pred["keypoints_scores"],
        ):
            if score > 0.8:
                # device -> host
                keypoints = keypoints.to("cpu").detach().numpy()
                keypoints_scores = keypoints_scores.to("cpu").detach().numpy()
                parts = keypoints[:3]  # nose, left_eye, right_eye
                parts_scores = keypoints_scores[:3]
                if (parts_scores > 8.0).all():
                    fontsize = int(np.linalg.norm(parts[0] - parts[1]) * 0.9)
                    if fontsize > 80:
                        continue
                    font = ImageFont.truetype("C:\Windows\Fonts\meiryob.ttc", fontsize)
                    for idx, (x, y, v) in enumerate(parts):
                        draw.text(
                            (x - fontsize // 2, y - fontsize // 2),
                            ("も", "の", "の")[idx],
                            font=font,
                            fill=(0, 0, 0),
                        )
                    cnt += 1
    return np.asarray(im)


def run(model, infile, outfile):
    probe = ffmpeg.probe(infile)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width, height = int(video_info["width"]), int(video_info["height"])
    proc_in = subprocess.Popen(
        ffmpeg.input(infile)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .compile(),
        stdout=subprocess.PIPE,
    )
    proc_out = subprocess.Popen(
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(width, height)
        )
        .output(outfile, pix_fmt="yuv420p")
        .overwrite_output()
        .compile(),
        stdin=subprocess.PIPE,
    )
    while True:
        inbytes = proc_in.stdout.read(width * height * 3)
        if len(inbytes) == 0:
            break
        inframe = np.frombuffer(inbytes, np.uint8).reshape((height, width, 3))
        outframe = process_nomono(model, inframe)
        proc_out.stdin.write(outframe.astype(np.uint8).tobytes())
    proc_in.wait()
    proc_out.stdin.close()
    proc_out.wait()


def main():
    global DEVICE, ALPHA, NROWS, NCOLS
    start_time = time.time()
    args = parser.parse_args()
    DEVICE = (
        args.device.lower()
        if args.device.lower() in ("cuda", "cpu")
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    torch.set_grad_enabled(False)
    run(model, args.infile, args.outfile)
    print(model)
    print("Time (s): {:.3f}".format(time.time() - start_time))


if __name__ == "__main__":
    main()
