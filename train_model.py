"""Entrenamiento reproducible para Spaceship Titanic."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPENDING = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
NUMERICAL = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',
             'TotalSpent','GroupSize','CabinNum']
CATEGORICAL = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side','NoSpending']

def engineer(frame, combined_counts):
    df=frame.copy()
    groups=df['PassengerId'].str.split('_').str[0]
    df['GroupSize']=groups.map(combined_counts).astype(float)
    cabin=df['Cabin'].str.split('/',expand=True)
    df['Deck']=cabin[0]
    df['CabinNum']=pd.to_numeric(cabin[1],errors='coerce')
    df['Side']=cabin[2]
    for col in SPENDING + ['Age']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    # No suponemos que un valor faltante sea cero: TotalSpent se deja faltante
    # cuando cualquier gasto está ausente; el imputador se ajusta solo en train.
    df['TotalSpent']=df[SPENDING].sum(axis=1,min_count=len(SPENDING))
    df['NoSpending']=pd.Series(np.where(df['TotalSpent']==0,'YES','NO'),index=df.index,dtype='object')
    df.loc[df[SPENDING].isna().any(axis=1),'NoSpending']=None
    for col in ['CryoSleep','VIP']:
        df[col]=df[col].astype('string').str.upper().map({'TRUE':'YES','FALSE':'NO'})
    return df[NUMERICAL+CATEGORICAL]

def make_model(estimator):
    prep=ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),
                         ('scaler',StandardScaler())]),NUMERICAL),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                         ('encoder',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL)
    ])
    return Pipeline([('preprocessing',prep),('classifier',estimator)])

def run(data_dir, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(data_dir/'train.csv', dtype={'PassengerId':'string'})
    test=pd.read_csv(data_dir/'test.csv', dtype={'PassengerId':'string'})
    template=pd.read_csv(data_dir/'sample_submission.csv', dtype={'PassengerId':'string'})
    if train['PassengerId'].duplicated().any() or test['PassengerId'].duplicated().any():
        raise ValueError('Hay identificadores duplicados')
    if not test['PassengerId'].reset_index(drop=True).equals(template['PassengerId'].reset_index(drop=True)):
        raise ValueError('test y sample_submission no tienen IDs en el mismo orden')
    if train['Transported'].isna().any():
        raise ValueError('Existen valores objetivo faltantes en train')
    group_counts=pd.concat([train['PassengerId'],test['PassengerId']]).str.split('_').str[0].value_counts()
    X=engineer(train,group_counts)
    X_test=engineer(test,group_counts)
    y=train['Transported'].astype(str).str.upper().map({'TRUE':1,'FALSE':0})
    if y.isna().any(): raise ValueError('Transported contiene valores no reconocidos')
    print('Train:',train.shape,' Test:',test.shape)
    print('Distribución target:',y.value_counts().to_dict())
    print('Faltantes por columna:\n',train.isna().sum().to_string())
    X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    models={
        'regresion_logistica':make_model(LogisticRegression(max_iter=1200,random_state=42)),
        'random_forest':make_model(RandomForestClassifier(n_estimators=250,min_samples_leaf=2,
                                                         n_jobs=-1,random_state=42))
    }
    metrics={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        predicted=model.predict(X_valid)
        metrics[name]={
            'accuracy':float(accuracy_score(y_valid,predicted)),
            'precision':float(precision_score(y_valid,predicted,zero_division=0)),
            'recall':float(recall_score(y_valid,predicted,zero_division=0)),
            'f1':float(f1_score(y_valid,predicted,zero_division=0))}
        print(f'\n{name}: {metrics[name]}')
    best_name=max(metrics,key=lambda name:metrics[name]['f1'])
    best=models[best_name]
    pred_valid=best.predict(X_valid)
    ConfusionMatrixDisplay.from_predictions(y_valid,pred_valid,display_labels=['No','Sí'],cmap='Blues')
    plt.title(f'Matriz de confusión - {best_name}')
    plt.tight_layout()
    plt.savefig(output_dir/'matriz_confusion.png',dpi=160)
    plt.close()
    print('\nModelo seleccionado:',best_name)
    print(classification_report(y_valid,pred_valid,zero_division=0))
    best.fit(X,y)
    predicted_test=best.predict(X_test).astype(bool)
    submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Transported':predicted_test})
    submission.to_csv(output_dir/'submission.csv',index=False)
    with open(output_dir/'metricas.json','w',encoding='utf-8') as f:
        json.dump({'modelos':metrics,'seleccionado':best_name,
                   'filas_train_pdf':len(train),'filas_test_pdf':len(test)},f,indent=2,ensure_ascii=False)
    print('Archivos generados en:',output_dir)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs'))
    args=p.parse_args()
    run(args.data_dir,args.output_dir)
if __name__=='__main__':main()
