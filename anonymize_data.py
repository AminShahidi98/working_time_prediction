"""
Make an anonymous copy of the thesis CSV exports.

    python anonymize_data.py --data ../Thesis_python/data_csv/ --out ../Thesis_python/data_csv_anon/

What changes:
  - machines     wc_name 'BRAND 100 X' -> 'Machine 01', wc_code 'T01' -> 'M01'
  - articles     ar_code '1234567'     -> 'ART-001'
  - work orders  wo_code '2024_100_10' -> 'WO-0001'

Everything else is copied exactly as it was. The same real code always gets
the same fake code in every file, so the files still join together. The fake
codes follow the sorted order of the real ones, so the pipeline groups and
sorts the orders as before and gives the same results.

The original folder is never touched. The key (real -> fake) is written next
to the output folder: keep it private and never commit it.
"""

import argparse
import os

import pandas as pd

SUBWORKINGS_CHUNK = 500_000


def read_text(path, columns=None):
    # every column as plain text, so the values are written back unchanged
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)


def clean(value):
    # the pipeline strips the codes, so ' 1234567  ' and '1234567' are the same article
    return value.strip()


def make_key(real_codes, prefix):
    real_codes = sorted(set(code for code in real_codes if code != ''))
    width = len(str(len(real_codes)))
    key = {}
    for number, code in enumerate(real_codes, start=1):
        key[code] = '%s-%0*d' % (prefix, width, number)
    return key


def replace(column, key):
    fake = []
    for value in column:
        code = clean(value)
        if code == '':
            fake.append('')     # empty stays empty: orphan records must stay orphans
        else:
            fake.append(key[code])
    return fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../Thesis_python/data_csv/')
    parser.add_argument('--out', default='../Thesis_python/data_csv_anon/')
    arguments = parser.parse_args()
    data = arguments.data
    out = arguments.out
    if os.path.abspath(data) == os.path.abspath(out):
        raise ValueError('the output folder must be different from the data folder')
    os.makedirs(out, exist_ok=True)

    workcenters = read_text(os.path.join(data, 'mes_workcenters.csv'))
    workorders = read_text(os.path.join(data, 'mes_workorders.csv'))
    test_rows = read_text(os.path.join(data, 'twin_test_rows.csv'))
    print('reading the codes of the subworkings (large file)...')
    sub_codes = read_text(os.path.join(data, 'scada_subworkings.csv'), ['wo_code', 'ar_code'])

    # --- one key per kind of name, shared by all files -----------------
    order_codes = []
    article_codes = []
    for table in [workorders, test_rows, sub_codes]:
        order_codes += [clean(code) for code in table['wo_code']]
        article_codes += [clean(code) for code in table['ar_code']]
    order_key = make_key(order_codes, 'WO')
    article_key = make_key(article_codes, 'ART')

    width = len(str(workcenters['wc_id'].astype(int).max()))
    machine_name = {}
    machine_code = {}
    for wc_id in workcenters['wc_id']:
        number = '%0*d' % (width, int(wc_id))
        machine_name[wc_id] = 'Machine ' + number
        machine_code[wc_id] = 'M' + number
    del sub_codes

    # --- small files -----------------------------------------------------
    real_machine_names = dict(zip(workcenters['wc_id'], workcenters['wc_name']))
    workcenters['wc_code'] = [machine_code[wc_id] for wc_id in workcenters['wc_id']]
    workcenters['wc_name'] = [machine_name[wc_id] for wc_id in workcenters['wc_id']]
    workcenters['wc_description'] = ''
    workcenters.to_csv(os.path.join(out, 'mes_workcenters.csv'), index=False)

    workorders['wo_code'] = replace(workorders['wo_code'], order_key)
    workorders['ar_code'] = replace(workorders['ar_code'], article_key)
    workorders.to_csv(os.path.join(out, 'mes_workorders.csv'), index=False)

    test_rows['wo_code'] = replace(test_rows['wo_code'], order_key)
    test_rows['ar_code'] = replace(test_rows['ar_code'], article_key)
    test_rows['machine_name'] = [machine_name[wc_id] for wc_id in test_rows['wc_id']]
    test_rows.to_csv(os.path.join(out, 'twin_test_rows.csv'), index=False)

    # the alarms hold no names: copied as they are
    alarms = read_text(os.path.join(data, 'scada_alarms.csv'))
    alarms.to_csv(os.path.join(out, 'scada_alarms.csv'), index=False)

    # --- the large file, a piece at a time ---------------------------------
    target = os.path.join(out, 'scada_subworkings.csv')
    chunks = pd.read_csv(os.path.join(data, 'scada_subworkings.csv'), dtype=str,
                         keep_default_na=False, chunksize=SUBWORKINGS_CHUNK)
    rows = 0
    for number, chunk in enumerate(chunks):
        chunk['wo_code'] = replace(chunk['wo_code'], order_key)
        chunk['ar_code'] = replace(chunk['ar_code'], article_key)
        chunk.to_csv(target, index=False, mode='w' if number == 0 else 'a', header=(number == 0))
        rows += len(chunk)
        print('  subworkings written:', format(rows, ','))

    # --- the key, outside the output folder ---------------------------------
    key_rows = []
    for real, fake in order_key.items():
        key_rows.append({'kind': 'work order', 'real': real, 'fake': fake})
    for real, fake in article_key.items():
        key_rows.append({'kind': 'article', 'real': real, 'fake': fake})
    for wc_id, fake in machine_name.items():
        key_rows.append({'kind': 'machine', 'real': real_machine_names[wc_id], 'fake': fake})
    key_path = os.path.join(os.path.dirname(os.path.abspath(out)),
                            'anonymization_key_PRIVATE.csv')
    pd.DataFrame(key_rows).to_csv(key_path, index=False)

    print()
    print('work orders: %d   articles: %d   machines: %d' % (
        len(order_key), len(article_key), len(machine_name)))
    print('anonymous files written to', os.path.abspath(out))
    print('private key written to', key_path)


if __name__ == '__main__':
    main()
