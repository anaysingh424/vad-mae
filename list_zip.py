import zipfile

zip_path = r"C:\Users\Anay\.gemini\antigravity\scratch\Avenue_Dataset.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Listing top-level and potential ground truth files/folders:")
    for name in zip_ref.namelist()[:50]:
        print(name)
    
    gt_files = [name for name in zip_ref.namelist() if 'ground_truth' in name.lower() or 'label' in name.lower()]
    print(f"\nFound {len(gt_files)} ground truth related files:")
    for gt in gt_files[:20]:
        print(gt)
