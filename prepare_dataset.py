import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Конфигурация
META_PATH = Path('HAM10000_metadata.csv')
IMG_DIRS = [Path('HAM10000_images_part_1'), Path('HAM10000_images_part_2')]
ROOT_DIR = Path('dataset')

def create_structure(df):
    # Физическое копирование файлов в структуру dataset/{split}/{class}
    for _, row in df.iterrows():
        target_dir = ROOT_DIR / row['split'] / row['dx']
        target_dir.mkdir(parents=True, exist_ok=True)
        
        src = Path(row['image_path'])
        dst = target_dir / f"{row['image_id']}.jpg"
        
        if src.exists():
            shutil.copy(src, dst)

def prepare_dataset():
    df = pd.read_csv(META_PATH)
    
    # Маппинг путей
    paths = {p.stem: str(p) for d in IMG_DIRS for p in d.glob('*.jpg')}
    df['image_path'] = df['image_id'].map(paths)
    df.dropna(subset=['image_path'], inplace=True)
    
    # Stratified Lesion-based Split (70/10/20)
    df_u = df.drop_duplicates('lesion_id')
    
    train_ids, temp_ids = train_test_split(
        df_u['lesion_id'], test_size=0.3, stratify=df_u['dx'], random_state=42
    )
    
    temp_df = df_u[df_u['lesion_id'].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_df['lesion_id'], test_size=2/3, stratify=temp_df['dx'], random_state=42
    )
    
    split_map = {**dict.fromkeys(train_ids, 'train'), 
                 **dict.fromkeys(val_ids, 'valid'), 
                 **dict.fromkeys(test_ids, 'test')}
    
    df['split'] = df['lesion_id'].map(split_map)
    
    # Создание папок и копирование
    if ROOT_DIR.exists():
        shutil.rmtree(ROOT_DIR)
    create_structure(df)
    
    df.to_csv('HAM10000_processed.csv', index=False)
    return df

if __name__ == '__main__':
    res = prepare_dataset()
    print("Dataset structure created successfully.")
    print(res.groupby(['split', 'dx']).size().unstack(0, fill_value=0))