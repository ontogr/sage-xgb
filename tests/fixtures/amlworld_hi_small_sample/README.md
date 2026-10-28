# AMLworld HI-Small sample

This directory holds a sample of the IBM AMLworld HI-Small Variant. The raw
data comes from the Kaggle dataset `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`.

The rows are real and unmodified. The files keep the names, header lines and
row text of the full data, so the same loader reads the sample and the full
data.

## Files

| File | Contents |
| --- | --- |
| `HI-Small_Trans.csv` | 28000 transactions in file order, 186 illicit |
| `HI-Small_accounts.csv` | The accounts that appear in the transactions |
| `HI-Small_Patterns.txt` | 47 laundering attempts from the Patterns file |
| `LICENSE` | The CDLA-Sharing-1.0 license text for the AMLworld data |

## Selection rule

The builder reads the full data and applies these steps in order.

1. A Pattern transaction line matches a raw row when all eleven columns are
   equal. A Pattern counts only when every one of its lines matches.
2. The two Patterns with the fewest transactions are taken for each of the
   eight typologies. More Patterns are added, smallest first, until each day
   block has at least 10 illicit transactions and the Tail
   has at least 3.
3. Every transaction between two Accounts of the chosen Patterns is kept.
4. Edge-case rows are added with every transaction of the Accounts involved:
   a send pair and a receive pair one hour apart, send pairs 1,440 and 10,080
   minutes apart on the same Account, two sends in one minute from one
   Account, and every transaction of an account number used under two banks.
5. 20 non-illicit Tail transactions are drawn at random.
6. Other transactions from days 1 to 10 are drawn at random, in proportion to
   the size of each day, until the file reaches 28000 rows.

The Account identity is the pair of bank and account number everywhere. The
accounts file writes bank ids without leading zeros, so the builder matches
its lines by the numeric bank id. The transactions file keeps the zeros.

## Counts per day block

Days 1 to 10 form the blocks below. The Tail is every row after day 10.

| Day block | Rows | Illicit rows |
| --- | --- | --- |
| 1-2 | 9787 | 28 |
| 3-4 | 2339 | 33 |
| 5-6 | 5520 | 45 |
| 7-8 | 5513 | 33 |
| 9-10 | 4802 | 29 |
| tail | 39 | 18 |

## Regenerating the sample

Run `make sample` from the repository root. The command reads the three raw
files from `data/` and rewrites this directory. It uses the seed 7. The
same raw files, seed and row target give byte-identical files.
