import argparse
import os
import json
from pathlib import Path

try:
    from ultralytics import YOLO, settings
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    print("Please ensure ultralytics package is installed (pip install ultralytics).")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models (especially YOLO26 m/l) on specified datasets")
    # 체크포인트 경로
    parser.add_argument("--models", nargs="+", default=["yolo26m.pt", "yolo26l.pt"], 
                        help="Paths to YOLO checkpoint files (.pt) to evaluate")
    # 데이터셋 YAML 경로
    parser.add_argument("--datasets", nargs="+", default=["coco128.yaml"], 
                        help="Paths to dataset YAML files (e.g., coco.yaml, coco128.yaml)")
    # 데이터셋 루트 디렉토리 (선택적)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory where datasets are stored/downloaded (updates ultralytics settings)")
    
    # 평가용 파라미터
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
    
    # 결과 저장 경로
    parser.add_argument("--save-dir", type=str, default="evaluation_results", 
                        help="Root directory to save evaluation results")
    
    args = parser.parse_args()
    
    # 데이터셋 최상위 경로 설정 (주어진 경우)
    if args.data_dir:
        data_dir_path = str(Path(args.data_dir).absolute())
        settings.update({'datasets_dir': data_dir_path})
        print(f"[*] Updated Ultralytics datasets_dir to: {data_dir_path}")

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
                
                # 복잡도 정보 추출 (Params, FLOPs)
                try:
                    info_tuple = model.info()
                    # model.info() typically returns (layers, params, gradients, flops)
                    if isinstance(info_tuple, tuple) and len(info_tuple) >= 4:
                        params = info_tuple[1]
                        flops = info_tuple[3]
                    else:
                        params, flops = "N/A", "N/A"
                except Exception:
                    params = flops = "N/A"
                
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
                precision = metrics.box.mp if hasattr(metrics.box, 'mp') else None
                recall = metrics.box.mr if hasattr(metrics.box, 'mr') else None
                inference_ms = float(metrics.speed['inference']) if 'inference' in metrics.speed else None
                fps = 1000.0 / inference_ms if inference_ms else None
                
                result_dict = {
                    "mAP50-95": float(map50_95),
                    "mAP50": float(map50),
                    "mAP75": float(map75),
                    "Precision": float(precision) if precision is not None else None,
                    "Recall": float(recall) if recall is not None else None,
                    "Params": params,
                    "FLOPs": flops,
                    "fitness": float(metrics.fitness),
                    "speed_inference_ms": inference_ms,
                    "FPS": fps
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
        
        # 5. 마크다운 리포트 생성 및 저장
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md_content = f"# 📊 YOLO Performance Evaluation Report\n\n"
        md_content += f"* **Date:** {now_str}\n"
        md_content += f"* **Dataset:** `{dataset_name}`\n"
        md_content += f"* **Environment:** Image Size {args.imgsz} | Batch Size {args.batch} | Device `{args.device}`\n\n"
        
        md_content += f"## 🏆 Performance Summary\n\n"
        md_content += f"| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Params | FLOPs(G) | Inference(ms) | FPS |\n"
        md_content += f"|---|---|---|---|---|---|---|---|---|\n"
        
        for m_name, res in dataset_results.items():
            if "error" in res:
                md_content += f"| {m_name} | ERROR | - | - | - | - | - | - | - |\n"
                continue
            
            p = f"{res.get('Precision', 0):.4f}" if res.get('Precision') is not None else "-"
            r = f"{res.get('Recall', 0):.4f}" if res.get('Recall') is not None else "-"
            map50 = f"{res.get('mAP50', 0):.4f}"
            map50_95 = f"{res.get('mAP50-95', 0):.4f}"
            params_val = res.get('Params', "-")
            flops_val = res.get('FLOPs', "-")
            inf = f"{res.get('speed_inference_ms', 0):.2f}" if res.get('speed_inference_ms') is not None else "-"
            fps_val = f"{res.get('FPS', 0):.1f}" if res.get('FPS') is not None else "-"
            
            md_content += f"| {m_name} | {p} | {r} | {map50} | {map50_95} | {params_val} | {flops_val} | {inf} | {fps_val} |\n"
            
        md_content += f"\n## 📝 Model Details\n\n"
        for m_name, res in dataset_results.items():
            if "error" in res:
                md_content += f"### 🔹 Model: {m_name}\n**Error:** {res['error']}\n\n"
                continue
            md_content += f"### 🔹 Model: {m_name}\n"
            md_content += f"* **mAP@0.75:** {res.get('mAP75', 0):.4f}\n"
            md_content += f"* **Complexity:** Params: {res.get('Params', '-')}, FLOPs(G): {res.get('FLOPs', '-')}\n"
            md_content += f"* **Speed:** Inference: {res.get('speed_inference_ms', '-')} ms, FPS: {res.get('FPS', '-')} Hz\n\n"
            
        md_file = save_base_dir / dataset_name / f"{dataset_name}_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Saved {dataset_name} Markdown report to {md_file}")
        
        # 6. CSV 요약본 생성 및 저장
        import csv
        csv_file = save_base_dir / dataset_name / f"{dataset_name}_summary.csv"
        csv_headers = ["Dataset", "Model", "Precision", "Recall", "mAP50", "mAP50-95", "mAP75", "Params", "FLOPs", "Inference_ms", "FPS"]
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            for m_name, res in dataset_results.items():
                if "error" in res:
                    writer.writerow({"Dataset": dataset_name, "Model": m_name, "Precision": "ERROR"})
                    continue
                writer.writerow({
                    "Dataset": dataset_name,
                    "Model": m_name,
                    "Precision": res.get("Precision"),
                    "Recall": res.get("Recall"),
                    "mAP50": res.get("mAP50"),
                    "mAP50-95": res.get("mAP50-95"),
                    "mAP75": res.get("mAP75"),
                    "Params": res.get("Params"),
                    "FLOPs": res.get("FLOPs"),
                    "Inference_ms": res.get("speed_inference_ms"),
                    "FPS": res.get("FPS"),
                })
        print(f"Saved {dataset_name} CSV summary to {csv_file}")
        
    # 전체 요약본 저장
    overall_summary_file = save_base_dir / "overall_summary.json"
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)
        
    print(f"\nAll evaluations complete. Overall summary saved to {overall_summary_file}")

if __name__ == "__main__":
    main()
