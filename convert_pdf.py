"""Reconstruye tablas de los PDF proporcionados por páginas y posición de texto."""
from pathlib import Path
import argparse
import fitz
import pandas as pd

COLUMNS = {
    'train': [['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','Age'],
              ['VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'],
              ['Name','Transported']],
    'test': [['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','Age'],
             ['VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'], ['Name']],
    'sample_submission': [['PassengerId','Transported']],
}

def read_part(page, headers, part_index, kind):
    rows = [[[] for _ in headers] for _ in range(45)]
    for word in page.get_text('words'):
        x, y, _, _, value, *_ = word
        if y < 70 or y > 785:
            continue
        idx = round((y - 73.6167) / 15.75)
        if idx < 0 or idx >= 45:
            continue
        while len(rows) <= idx:
            rows.append([[] for _ in headers])
        if kind == 'sample_submission':
            column = 0 if x < 130 else 1
        elif part_index == 2:
            column = 0 if x < 135 else 1
        else:
            # En estos PDF, el texto numérico aparece alineado a la derecha.
            # Fronteras verticales de las columnas originales.
            boundaries = [115, 201, 277, 352, 428]
            column = sum(x >= edge for edge in boundaries)
        if column < len(headers):
            rows[idx][column].append((x, value))
    result = []
    for cells in rows:
        result.append({h: ' '.join(v for _,v in sorted(c)).strip() or None
                       for h,c in zip(headers,cells)})
    return result

def convert(source, out, kind):
    doc = fitz.open(source)
    parts = COLUMNS[kind]
    if len(doc) % len(parts):
        raise ValueError(f'Cantidad de páginas incompatible: {source}')
    all_rows=[]
    for offset in range(0,len(doc),len(parts)):
        extracted = [read_part(doc[offset+i],h,i,kind) for i,h in enumerate(parts)]
        counts = [len(x) for x in extracted]
        if len(set(counts)) != 1:
            raise ValueError(f'Páginas no alineadas en {offset}: {counts}')
        for trio in zip(*extracted):
            merged={}
            for obj in trio: merged.update(obj)
            if merged.get('PassengerId'):
                all_rows.append(merged)
    data = pd.DataFrame(all_rows)
    if data['PassengerId'].duplicated().any():
        raise ValueError('PassengerId duplicados; revisar extracción PDF')
    if not data['PassengerId'].str.fullmatch(r'\d{4}_\d{2}').all():
        raise ValueError('Formato inválido de identificador; revisar extracción PDF')
    for col in ['CryoSleep','VIP','Transported']:
        if col in data:
            values=set(data[col].dropna().unique())
            if not values <= {'TRUE','FALSE'}: raise ValueError(f'{col}: {values}')
    out.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(out,index=False)
    print(f'{kind}: {len(data):,} registros, {len(data.columns)} columnas -> {out}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf-dir', type=Path, default=Path('.'))
    parser.add_argument('--out-dir', type=Path, default=Path('data'))
    args=parser.parse_args()
    for kind in COLUMNS:
        convert(args.pdf_dir / f'{kind}.pdf', args.out_dir / f'{kind}.csv', kind)
if __name__=='__main__': main()
