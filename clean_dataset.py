import json, re, pathlib

f = pathlib.Path('dataset.jsonl')
if not f.exists():
    print('No dataset.jsonl found')
    exit()

sentence_end = re.compile(r'[.!?]\s*$')
lines = f.read_text(encoding='utf-8').splitlines()

good, bad = [], 0
for l in lines:
    if not l.strip():
        continue
    r = json.loads(l)
    cap = r.get('caption', '').strip()
    if cap and sentence_end.search(cap):
        good.append(l)
    else:
        bad += 1
        print(f"  REMOVING: {r['image']}  caption={cap[:60]!r}")

f.write_text('\n'.join(good) + ('\n' if good else ''), encoding='utf-8')
print(f'\nRemoved {bad} bad records. Kept {len(good)} good records.')
