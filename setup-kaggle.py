from pathlib import Path
import shutil

def setup_dataset():
    """Setup dataset from your downloaded folders"""
    
    print("="*60)
    print("EyeMRI AI - Dataset Setup")
    print("="*60)
    
    # Check which dataset exists
    datasets = []
    
    diagnosis_path = Path('Diagnosis of Diabetic Retinopathy')
    retino_path = Path('retino')
    
    if diagnosis_path.exists():
        datasets.append(('Diagnosis of Diabetic Retinopathy', diagnosis_path))
        print(f"✅ Found: Diagnosis of Diabetic Retinopathy/")
    
    if retino_path.exists():
        datasets.append(('retino', retino_path))
        print(f"✅ Found: retino/")
    
    if not datasets:
        print("\n❌ No dataset found!")
        print("\nExpected structure:")
        print("  Diagnosis of Diabetic Retinopathy/")
        print("    ├── train/")
        print("    │   ├── dr/")
        print("    │   └── no_dr/")
        print("    ├── test/")
        print("    └── valid/")
        print("\nOR")
        print("  retino/")
        print("    ├── train/")
        print("    ├── test/")
        print("    └── valid/")
        return False
    
    print(f"\nFound {len(datasets)} dataset(s)")
    
    # Choose which dataset to use
    if len(datasets) > 1:
        print("\nMultiple datasets found. Choose one:")
        for i, (name, _) in enumerate(datasets, 1):
            print(f"{i}. {name}")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        dataset_idx = int(choice) - 1 if choice in ['1', '2'] else 0
    else:
        dataset_idx = 0
    
    dataset_name, dataset_path = datasets[dataset_idx]
    
    print(f"\n📂 Using dataset: {dataset_name}")
    
    # Count images
    stats = count_images(dataset_path)
    
    if stats:
        print_statistics(stats)
        
        # Create symbolic link or copy to standard location
        setup_standard_structure(dataset_path, dataset_name)
        
        return True
    
    return False

def count_images(dataset_path):
    """Count images in each split"""
    
    stats = {}
    
    for split in ['train', 'test', 'valid']:
        split_path = dataset_path / split
        
        if not split_path.exists():
            continue
        
        dr_path = split_path / 'dr'
        no_dr_path = split_path / 'no_dr'
        
        dr_images = len(list(dr_path.glob('*.jpg'))) + len(list(dr_path.glob('*.png'))) if dr_path.exists() else 0
        no_dr_images = len(list(no_dr_path.glob('*.jpg'))) + len(list(no_dr_path.glob('*.png'))) if no_dr_path.exists() else 0
        
        stats[split] = {
            'dr': dr_images,
            'no_dr': no_dr_images,
            'total': dr_images + no_dr_images
        }
    
    return stats

def print_statistics(stats):
    """Print dataset statistics"""
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    total_all = 0
    
    for split, counts in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  DR images:     {counts['dr']:4d}")
        print(f"  No DR images:  {counts['no_dr']:4d}")
        print(f"  Total:         {counts['total']:4d}")
        total_all += counts['total']
    
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES: {total_all}")
    print(f"{'='*60}")

def setup_standard_structure(dataset_path, dataset_name):
    """Create standard data/ folder structure"""
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create symbolic link or note the path
    link_path = data_dir / 'dataset'
    
    # Save dataset path for other scripts
    config = {
        'dataset_name': dataset_name,
        'dataset_path': str(dataset_path.absolute()),
        'structure': 'folder_based',
        'classes': ['no_dr', 'dr'],
        'splits': ['train', 'test', 'valid']
    }
    
    import json
    with open(data_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to: data/config.json")
    print(f"✅ Dataset ready to use!")
    
    return True

def verify_structure(dataset_path):
    """Verify folder structure is correct"""
    
    print("\n🔍 Verifying structure...")
    
    required = ['train', 'test', 'valid']
    missing = []
    
    for split in required:
        split_path = dataset_path / split
        if not split_path.exists():
            missing.append(split)
            continue
        
        dr_path = split_path / 'dr'
        no_dr_path = split_path / 'no_dr'
        
        if not dr_path.exists():
            missing.append(f"{split}/dr")
        if not no_dr_path.exists():
            missing.append(f"{split}/no_dr")
    
    if missing:
        print(f"⚠️  Missing folders: {', '.join(missing)}")
        return False
    
    print("✅ All folders present!")
    return True

if __name__ == "__main__":
    success = setup_dataset()
    
    if success:
        print("\n" + "="*60)
        print("✅ Dataset Setup Complete!")
        print("="*60)
        print("\nNext step: Run 3_preprocess_data.py")
    else:
        print("\n" + "="*60)
        print("❌ Setup Failed")
        print("="*60)
        print("\nPlease ensure your dataset is extracted and in the correct location.")