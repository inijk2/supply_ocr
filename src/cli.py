from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supply/selection/queue OCR pipeline (480p).")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--output", default="output.json", help="Output JSON path")
    parser.add_argument("--profile", default=str(Path(__file__).resolve().parent / "roi" / "profile_480p.json"))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=420.0)
    parser.add_argument("--ocr", default=None, help="OCR engine: paddleocr|easyocr|tesseract|auto")
    parser.add_argument("--fps", type=float, default=2.0, help="Supply sampling FPS")
    parser.add_argument("--supply-samples", type=int, default=7, help="Frames per supply window")
    parser.add_argument("--roi-samples", type=int, default=10, help="Frames per ROI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        video_path=args.video,
        profile_path=args.profile,
        output_path=args.output,
        start_sec=args.start,
        end_sec=args.end,
        ocr_engine=args.ocr,
        supply_fps=args.fps,
        supply_samples=args.supply_samples,
        roi_samples=args.roi_samples,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
