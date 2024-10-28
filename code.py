import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Titanic_Dataset.csv'
titanic_data = pd.read_csv(file_path)

titanic_data = titanic_data.drop_duplicates()

def infer_sex_from_title(name):
    if "Mr." in name:
        return "male"
    elif "Mrs." in name or "Miss" in name or "Ms." in name:
        return "female"
    elif "Master" in name:
        return "male"
    else:
        return "unknown" 

titanic_data['sex'] = titanic_data.apply(
    lambda row: infer_sex_from_title(row['name']) if row['sex'] == "unknown" else row['sex'],
    axis=1
)
titanic_data['fare'].fillna(titanic_data['fare'].median(), inplace=True)
titanic_data['embarked'].fillna(titanic_data['embarked'].mode()[0], inplace=True)


cols_to_impute = ['pclass', 'age', 'sibsp', 'parch', 'fare']
knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
titanic_data_numeric = titanic_data[cols_to_impute]

titanic_data_imputed = knn_imputer.fit_transform(titanic_data_numeric)
titanic_data['age'] = titanic_data_imputed[:, 1]

age_mean = titanic_data.loc[(titanic_data['age'] >= 0) & (titanic_data['age'] <= 90), 'age'].mean()
titanic_data['age'] = titanic_data['age'].apply(lambda x: age_mean if x < 0 or x > 90 else x)
titanic_data['age'] = titanic_data['age'].round().astype(int)

missing_final = titanic_data.isnull().sum()
duplicates_final = titanic_data.duplicated().sum()


#titanic_data.to_csv('Titanic_Dataset_cleaned.csv', index=False)


def annotate_barplot(ax, percentages):
    for p, percentage in zip(ax.patches, percentages):
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom')

overall_survival_rate = titanic_data['survived'].mean() * 100
print(f"Tasso di sopravvivenza complessivo: {overall_survival_rate:.2f}%")

survival_by_sex = titanic_data.groupby('sex')['survived'].mean() * 100
print("\nTasso di sopravvivenza per sesso:")
print(survival_by_sex)

plt.figure(figsize=(8, 6))
ax = sns.barplot(x='sex', y='survived', data=titanic_data, ci=None)
plt.title('Tasso di sopravvivenza per sesso')
plt.ylabel('Tasso di sopravvivenza')
plt.xlabel('sesso')
annotate_barplot(ax, survival_by_sex.values)
plt.show()

survival_by_class = titanic_data.groupby('pclass')['survived'].mean() * 100
print("\nTasso di sopravvivenza per classe di viaggio:")
print(survival_by_class)

plt.figure(figsize=(8, 6))
ax = sns.barplot(x='pclass', y='survived', data=titanic_data, ci=None)
plt.title('Tasso di sopravvivenza per classe di viaggio')
plt.ylabel('Tasso di sopravvivenza')
plt.xlabel('classe')
annotate_barplot(ax, survival_by_class.values)
plt.show()

survival_by_embarked = titanic_data.groupby('embarked')['survived'].mean() * 100
print("\nTasso di sopravvivenza per porto di imbarco:")
print(survival_by_embarked)

survival_by_embarked = titanic_data.groupby('embarked')['survived'].mean() * 100
survival_by_embarked = survival_by_embarked.reset_index()

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=survival_by_embarked, x='embarked', y='survived')
plt.title('Tasso di sopravvivenza per imbarco')
plt.xlabel('Imbarco')
plt.ylabel('Tasso di sopravvivenza (%)')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom')

plt.ylim(0, 100)  
plt.show()

titanic_data['age_group'] = pd.cut(titanic_data['age'], bins=[0, 12, 18, 35, 60, 90], 
                                   labels=['Bambini', 'Adolescenti', 'Giovani Adulti', 'Adulti', 'Anziani'])

survival_by_age_group = titanic_data.groupby('age_group')['survived'].mean() * 100
print("\nTasso di sopravvivenza per fascia di età:")
print(survival_by_age_group)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='age_group', y='survived', data=titanic_data, ci=None)
plt.title('Tasso di sopravvivenza per fascia di età')
plt.ylabel('Tasso di sopravvivenza')
annotate_barplot(ax, survival_by_age_group.values)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data=titanic_data, x='age', hue='survived', kde=True, bins=30, palette='crest')
plt.title('Distribuzione dell\'età dei passeggeri per sopravvivenza')
plt.xlabel('Età')
plt.ylabel('sopravvissuti')
plt.legend(title='Survived', labels=['Non Sopravvissuti', 'Sopravvissuti'])
plt.show()

titanic_data['age_group'] = pd.cut(titanic_data['age'], bins=range(0, 91, 5), right=False)

survival_by_age_sex = titanic_data.groupby(['age_group', 'sex'])['survived'].mean().reset_index()

plt.figure(figsize=(15, 8))
sns.barplot(data=survival_by_age_sex, x='age_group', y='survived', hue='sex', 
            palette={'male': '#a8f5f7', 'female': 'pink'})
plt.title('Tasso di sopravvivenza per fasce di età (5 anni) e sesso')
plt.xlabel('Fasce di età')
plt.ylabel('Tasso di sopravvivenza')
plt.legend(title='Sesso', loc='upper right')
plt.xticks(rotation=90) 
plt.show()