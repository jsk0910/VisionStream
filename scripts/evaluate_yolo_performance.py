import argparse
import os
import json
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    print("Please ensure ultralytics package is installed (pip install ultralytics).")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models (especially YOLO26 m/l) on specified datasets")
    # 체크포인트 경로 (기본값 설정)
    parser.add_argument("--models", nargs="+", default=["yolo26m.pt", "yolo26l.pt"], 
                        help="Paths to YOLO checkpoint files (.pt) to evaluate")
    # 데이터셋 경로 (기본값 설정)
    parser.add_argument("--datasets", nargs="+", default=["coco128.yaml"], 
                        help="Paths to dataset YAML files (e.g., coco.yaml, coco128.yaml)")
    
    # 평가용 파라미터
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
    
    # 결과 저장 경로
    parser.add_argument("--save-dir", type=str, default="evaluation_results", 
                        help="Root directory to save evaluation results")
    
    args = parser.parse_args()
    
    save_base_dir = Path(args.save_dir)
    save_base_dir.mkdir(parents=True, exist_ok=True)
    
    summary_results = {}
    
    # 각 데이터셋별로 평가 진행
    for dataset in args.datasets:
        dataset_name = Path(dataset).stem
        print(f"\n{'='*50}")
        print(f"Evaluating on dataset: {dataset_name} ({dataset})")
        print(f"{'='*50}\n")
        
        dataset_results = {}
        
        # 해당 데이터셋에 대해 지정된 각 모델(체크포인트) 평가
        for model_path in args.models:
            model_name = Path(model_path).stem
            print(f"\n--- Model: {model_name} ({model_path}) ---")
            
            try:
                # 1. 체크포인트 로딩
                print(f"Loading checkpoint from: {model_path}")
                model = YOLO(model_path)
                
                # 2. 성능 측정 및 데이터 로딩
                print(f"Starting validation for {model_name} on {dataset_name}...")
                
                # 결과를 저장할 프로젝트 디렉토리: <save_dir>/<dataset_name>/<model_name>
                project_dir = save_base_dir / dataset_name
                name_dir = model_name
                
                # val() 함수로 검증 수행 (데이터 로딩, 평가 자동 수행)
                metrics = model.val(
                    data=dataset,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    project=str(project_dir),
                    name=name_dir,
                    exist_ok=True  # 결과 덮어쓰기 허용 (기존 디렉토리 재사용)
                )
                
                # 3. 측정된 결과(mAP, 속도 등) 추출
                map50_95 = metrics.box.map    # mAP@0.5:0.95
                map50 = metrics.box.map50     # mAP@0.5
                map75 = metrics.box.map75     # mAP@0.75
                
                result_dict = {
                    "mAP50-95": float(map50_95),
                    "mAP50": float(map50),
                    "mAP75": float(map75),
                    "fitness": float(metrics.fitness),
                    "speed_inference_ms": float(metrics.speed['inference']) if 'inference' in metrics.speed else None
                }
                
                dataset_results[model_name] = result_dict
                
                print(f"\nResults for {model_name} on {dataset_name}:")
                for k, v in result_dict.items():
                    print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")
                    
            except Exception as e:
                print(f"[-] Error evaluating {model_name} on {dataset_name}: {e}")
                dataset_results[model_name] = {"error": str(e)}
                
        summary_results[dataset_name] = dataset_results
        
        # 4. 측정한 결과를 개별 데이터셋별 JSON으로 정리하여 저장
        dataset_summary_file = save_base_dir / dataset_name / f"{dataset_name}_summary.json"
        dataset_summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_summary_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=4, ensure_ascii=False)
        print(f"\nSaved {dataset_name} detailed summary to {dataset_summary_file}")
        
    # 전체 요약본 저장
    overall_summary_file = save_base_dir / "overall_summary.json"
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)
        
    print(f"\nAll evaluations complete. Overall summary saved to {overall_summary_file}")

if __name__ == "__main__":
    main()
