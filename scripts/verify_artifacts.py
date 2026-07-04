import json
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    
    if str(filepath).endswith(".json"):
        # For JSON files, normalize line endings to LF to prevent cross-platform CI mismatch
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().replace('\r\n', '\n')
            sha256_hash.update(content.encode("utf-8"))
    else:
        # Binary files like .joblib
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
    return sha256_hash.hexdigest()

def generate_manifest():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    
    files_to_hash = {
        "final_pipeline.joblib": artifacts_dir / "model" / "final_pipeline.joblib",
        "decision_threshold.json": artifacts_dir / "model" / "decision_threshold.json",
        "model_metadata.json": artifacts_dir / "model" / "model_metadata.json",
    }
    
    manifest = {}
    for name, path in files_to_hash.items():
        if path.exists():
            manifest[name] = get_sha256(path)
        else:
            print(f"Warning: {path} not found.")
            
    with open(artifacts_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    print("Manifest generated successfully.")

def verify_artifacts():
    manifest_path = PROJECT_ROOT / "artifacts" / "manifest.json"
    if not manifest_path.exists():
        print("Manifest not found! Generating one now...")
        generate_manifest()
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    artifacts_dir = PROJECT_ROOT / "artifacts"
    files_to_hash = {
        "final_pipeline.joblib": artifacts_dir / "model" / "final_pipeline.joblib",
        "decision_threshold.json": artifacts_dir / "model" / "decision_threshold.json",
        "model_metadata.json": artifacts_dir / "model" / "model_metadata.json",
    }
    
    all_valid = True
    for name, expected_hash in manifest.items():
        path = files_to_hash.get(name)
        if not path or not path.exists():
            print(f"✗ Artifact missing: {name}")
            all_valid = False
            continue
            
        actual_hash = get_sha256(path)
        if actual_hash == expected_hash:
            print(f"PASS: {name} valid.")
        else:
            print(f"FAIL: {name} checksum mismatch!")
            all_valid = False
            
    if not all_valid:
        print("Artifact verification failed!")
        sys.exit(1)
    else:
        print("PASS: All artifacts verified successfully.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_manifest()
    else:
        verify_artifacts()
