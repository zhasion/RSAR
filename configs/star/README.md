The files in this folder are the config files of the two models, `Deformable DETR` and `ARS-DETR`. 

Before using them, you need to utilize the [STAR-MMRotate](https://github.com/yangxue0827/STAR-MMRotate) repository and move the contents of the folder according to the following relationship.

```sh
star/configs/rsar.py			                ->  STAR-MMRotate/configs/_base_/datasets/rsar.py
configs/star/datasets/rsar.py             ->  STAR-MMRotate/mmrotate/datasets/rsar.py
configs/star/csl_detr_r50_rsar.py         ->  STAR-MMRotate/configs/ars_detr/csl_detr_r50_rsar.py
configs/star/deformable_detr_r50_rsar.py  ->  STAR-MMRotate/configs/ars_detr/deformable_detr_r50_rsar.py
configs/star/arcsl_detr_r50_rsar.py       ->  STAR-MMRotate/configs/ars_detr/arcsl_detr_r50_rsar.py
```

Finally, you need to add the import of RSARDataset in `STAR-MMRotate/mmrotate/datasets/__init__.py`

```python
from .rsar import RSARDataset
```

