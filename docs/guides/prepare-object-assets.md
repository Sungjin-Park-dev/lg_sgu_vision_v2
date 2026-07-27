# �� ��ü �ڻ� �غ�

���� Object ��Ͽ� ǥ�õǷ��� `data/{object}/mesh/source.obj`�� �ʿ��ϴ�. Isaac Pipeline���� �ҷ������� `source.usd`�� �غ��Ѵ�.

## 1. OBJ ����ȭ

```bash
uv run scripts/setup/prepare_object_mesh.py normalize \
  --object my_object --input /path/to/raw.obj
```

�⺻������ mm�� m�� ��ȯ�ϰ� �ٴ� �߽��� ������ �����. ����� ���� �ʰ� Ȯ���Ϸ��� `--dry-run`�� ����Ѵ�.

## 2. ���� ����

```bash
uv run scripts/setup/prepare_object_mesh.py reorient \
  --object my_object --euler 90 0 0
```

Euler ��� `--quat W X Y Z` �Ǵ� `--world-target-quat W X Y Z`�� ����� �� �ִ�. ���� ������ ��� ���� �⺻������ ����� �����.

## 3. USD ����

```bash
uv run scripts/setup/build_object_usd.py --object my_object --force
```

ī�޶� CAD �Ǵ� preview ghost�� ��ü�� ���� ���� ������ ����Ѵ�.

```bash
uv run scripts/setup/build_camera_mesh.py --source /path/to/camera.obj --dry-run
uv run scripts/setup/build_ghost_usd.py
```

�޽ø� �ٲٰų� ũ�� ȸ���ߴٸ� ������Ʈ�� �ٽ� �����Ѵ�.
