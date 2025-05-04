import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

# Расширенные свойства аминокислот с добавлением степени свободы
AA_PROPERTIES = {
    'A': {'hydrophobic': 1.80, 'size': 0.3, 'charge': 0, 'flexibility': 0.2},
    'C': {'hydrophobic': 2.50, 'size': 0.4, 'charge': 0, 'flexibility': 0.3},
    'D': {'hydrophobic': -3.50, 'size': 0.5, 'charge': -1, 'flexibility': 0.5},
    'E': {'hydrophobic': -3.50, 'size': 0.6, 'charge': -1, 'flexibility': 0.7},
    'F': {'hydrophobic': 2.80, 'size': 0.7, 'charge': 0, 'flexibility': 0.4},
    'G': {'hydrophobic': -0.40, 'size': 0.1, 'charge': 0, 'flexibility': 0.9},
    'H': {'hydrophobic': -3.20, 'size': 0.6, 'charge': 0.5, 'flexibility': 0.5},
    'I': {'hydrophobic': 4.50, 'size': 0.6, 'charge': 0, 'flexibility': 0.3},
    'K': {'hydrophobic': -3.90, 'size': 0.6, 'charge': 1, 'flexibility': 0.8},
    'L': {'hydrophobic': 3.80, 'size': 0.6, 'charge': 0, 'flexibility': 0.4},
    'M': {'hydrophobic': 1.90, 'size': 0.6, 'charge': 0, 'flexibility': 0.7},
    'N': {'hydrophobic': -3.50, 'size': 0.5, 'charge': 0, 'flexibility': 0.6},
    'P': {'hydrophobic': -1.60, 'size': 0.4, 'charge': 0, 'flexibility': 0.1},
    'Q': {'hydrophobic': -3.50, 'size': 0.6, 'charge': 0, 'flexibility': 0.7},
    'R': {'hydrophobic': -4.50, 'size': 0.7, 'charge': 1, 'flexibility': 0.8},
    'S': {'hydrophobic': -0.80, 'size': 0.4, 'charge': 0, 'flexibility': 0.6},
    'T': {'hydrophobic': -0.70, 'size': 0.5, 'charge': 0, 'flexibility': 0.4},
    'V': {'hydrophobic': 4.20, 'size': 0.5, 'charge': 0, 'flexibility': 0.2},
    'W': {'hydrophobic': -0.90, 'size': 0.8, 'charge': 0, 'flexibility': 0.3},
    'Y': {'hydrophobic': -1.30, 'size': 0.7, 'charge': 0, 'flexibility': 0.4}
}

# Определяем значения для категорий местоположения мутаций
LOCATION_CATEGORIES = ['COR', 'INT', 'SUP', 'RIM', 'SUR']

def parse_single_mutation(mutation_str):
    """
    Парсит строку одиночной мутации формата 'IA96A'
    """
    if pd.isna(mutation_str) or ',' in mutation_str:
        return None
    
    pattern = r'([A-Z])([A-Z])(\d+)([A-Z])'
    match = re.match(pattern, mutation_str.strip())
    
    if match:
        chain, wild_type, position, mutant = match.groups()
        return (chain, wild_type, int(position), mutant)
    
    return None

def extract_features(row):
    """
    Извлекает признаки из строки данных
    """
    mutation_info = parse_single_mutation(row['Mutation(s)_cleaned'])
    
    if mutation_info is None:
        return None
    
    chain, wild_type, position, mutant = mutation_info
    
    features = {}
    
    if wild_type not in AA_PROPERTIES or mutant not in AA_PROPERTIES:
        return None
    
    wt_props = AA_PROPERTIES[wild_type]
    mut_props = AA_PROPERTIES[mutant]
    
    # Свойства дикого типа
    features['wt_hydrophobic'] = wt_props['hydrophobic']
    features['wt_size'] = wt_props['size']
    features['wt_charge'] = wt_props['charge']
    features['wt_flexibility'] = wt_props['flexibility']
    
    # Свойства мутанта
    features['mut_hydrophobic'] = mut_props['hydrophobic']
    features['mut_size'] = mut_props['size']
    features['mut_charge'] = mut_props['charge']
    features['mut_flexibility'] = mut_props['flexibility']
    
    # Изменения свойств
    features['delta_hydrophobic'] = mut_props['hydrophobic'] - wt_props['hydrophobic']
    features['delta_size'] = mut_props['size'] - wt_props['size']
    features['delta_charge'] = mut_props['charge'] - wt_props['charge']
    features['delta_flexibility'] = mut_props['flexibility'] - wt_props['flexibility']
    
    # Местоположение мутации
    location = row['iMutation_Location(s)']
    if pd.isna(location) or location not in LOCATION_CATEGORIES:
        return None
    
    features['mutation_location'] = location
    
    # Целевая переменная
    features['ddG_sign'] = 1 if row['ddG'] > 0 else 0
    features['ddG_value'] = row['ddG']
    
    return features

def create_feature_dataset(df):
    """
    Создает датасет с признаками для машинного обучения
    """
    features_list = []
    
    for idx, row in df.iterrows():
        features = extract_features(row)
        
        if features is None:
            continue
            
        features['index'] = idx
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    if len(features_df) == 0:
        raise ValueError("Не удалось извлечь признаки ни из одной строки")
    
    # Факторизуем местоположение мутации
    le = LabelEncoder()
    features_df['mutation_location_encoded'] = le.fit_transform(features_df['mutation_location'])
    
    # Сохраняем маппинг для интерпретации
    location_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Кодирование местоположений:")
    for loc, code in location_mapping.items():
        print(f"{loc}: {code}")
    
    # Удаляем текстовое представление
    features_df = features_df.drop('mutation_location', axis=1)
    
    # Перемещаем index в начало
    cols = ['index'] + [col for col in features_df.columns if col != 'index']
    features_df = features_df[cols]
    
    return features_df, le

# Загружаем данные
df = pd.read_csv('cleaned_with_ddG.csv')

# Создаем датасет с признаками
features_df, location_encoder = create_feature_dataset(df)

# Сохраняем результат
features_df.to_csv('single_mutations_features_factorized.csv', index=False)

# Сохраняем encoder для использования с новыми данными
import joblib
joblib.dump(location_encoder, 'location_encoder.joblib')

print(f"Создан датасет с {features_df.shape[0]} строками и {features_df.shape[1]} столбцами")

# Создаем файлы для обучения
X = features_df.drop(['ddG_sign', 'ddG_value'], axis=1)
y = features_df['ddG_sign']

X.to_csv('X_features_factorized.csv', index=False)
y.to_csv('y_target.csv', index=False)

print("\nФайлы для обучения модели сохранены:")
print("- X_features_factorized.csv (признаки)")
print("- y_target.csv (целевая переменная)")
print(f"Размерность признаков: {X.shape}")