# Add a Variant

A Variant is one AMLworld dataset, with its own config file, day boundaries and raw files. Only HI-Small is tested. A new Variant needs measured day boundaries in its own config, and it may need new currency rates or payment formats in the code.

## Prerequisites

- The Variant's Transactions CSV, named `<name>_Trans.csv`, in the `data.raw_dir` folder. The loader reads the AMLworld column layout, with the header line first.
- A clone with `make sync` done.

## Measure the days

The split uses whole days. Count the Transactions on each day of the Variant, and choose the day boundaries from those counts.

```python
from pathlib import Path

import polars as pl

from sage_xgb.data.load import scan_raw

daily = (
    scan_raw(Path("data"), "<name>")
    .select(pl.col("timestamp").dt.date().alias("date"))
    .group_by("date")
    .len()
    .sort("date")
    .collect()
)
print(daily)
```

Replace `<name>` with the Variant name, such as the one in `data.name`. Choose `train_end`, `val_end` and `test_end` so that the train, validation and test splits hold about 60, 20 and 20 percent of the rows. Each boundary is the first day of the next split. The Tail starts on `test_end`.

## Write the config

Create `configs/<variant>.yaml` with the measured dates. The `extends` key loads `configs/hi_small.yaml` first and applies the new values on top.

```yaml
extends: hi_small.yaml

variant: <variant>

data:
  name: <Name>
  raw_dir: data

split:
  window_start: <YYYY-MM-DD>
  train_end: <YYYY-MM-DD>
  val_end: <YYYY-MM-DD>
  test_end: <YYYY-MM-DD>
  crossfit_blocks: [[1, 2], [3, 4], [5, 6]]
```

Replace every placeholder. A placeholder date fails the config check when the file loads. The `crossfit_blocks` value must list the training days from 1 to the number of days between `window_start` and `train_end`, each once and in order. The blocks must be at least two in number. Adjust the blocks when the training window has a different length.

Do not reuse the HI-Small dates. Each Variant has its own day boundaries.

## Add the currencies and payment formats

The pipeline checks that every paid currency, received currency and payment format is in the fixed vocabulary in `sage_xgb/data/vocab.py`. A value that is not listed stops the prepare stage. The error names the value and the column.

To add a value, add a currency rate to `USD_PER_UNIT`, or add a payment format to `PAYMENT_FORMATS`. The rates are fixed numbers. The code does not fit them from the Transactions, so a new rate must be a constant that you record in the code.

Currency codes are positions in the sorted currency names, and payment format codes are positions in `PAYMENT_FORMATS`. Adding a value can shift the codes of the values after it. Stage manifests do not hash `vocab.py`. After you change a rate or the format list, rerun the affected stages with `--from-stage prepare`. The run would otherwise reuse outputs that the old vocabulary produced.

## Run the Variant

Run the pipeline with the new config.

```bash
uv run sage-xgb run --config configs/<variant>.yaml
```

The run writes its outputs under `artifacts/<variant>/`. Run `sage-xgb report --config configs/<variant>.yaml` once the training stage has finished. The report reads the test and Tail labels, as described in [Test labels are read once](../explanation/split-and-tail.md#test-labels-are-read-once).

## Download the files

The `download` command fetches the files named `<name>_Trans.csv`, `<name>_accounts.csv` and `<name>_Patterns.txt` from the Kaggle dataset in `data.kaggle_dataset`. Set `data.kaggle_dataset` to the dataset that holds your Variant. The pipeline reads only the Transactions file.

## Check the result

The `prepare` stage raises an error when a Transaction falls before `window_start`. Use the day count from the measuring step to confirm that the first day of the file is the day that `window_start` names.
