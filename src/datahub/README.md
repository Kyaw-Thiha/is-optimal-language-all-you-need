# DataHub Package

This package houses the data-facing building blocks for the project—everything you need to download, normalize, and reload multilingual sense-evaluation corpora in a unified shape.

## Modules

- `download.py`: Pulls XL-WSD, XL-WiC, and MCL-WiC from Hugging Face and mirrors them under `data/raw/{dataset}`.
- `preprocess.py`: Converts each raw dataset into the shared `SenseSample` representation, saving splits to `data/preprocess/{dataset}/{split}`.
- `helpers.py`: Shared utilities (e.g., robust type coercions, mapping guards) used across download/preprocess/load flows.
- `loader.py`: Reads preprocessed splits from disk and yields `SenseSample` objects for downstream metrics.
- `sense_sample.py`: Defines the `SenseSample` dataclass—our minimal, task-agnostic view of a sense instance.

## SenseSample Schema

```python
@dataclass(frozen=True)
class SenseSample:
    sample_id: str
    dataset_id: str
    split: str
    language: str
    text_a: str
    text_b: Optional[str]
    lemma: str
    target_span: Optional[Tuple[int, int]]
    sense_tag: Optional[str]
    same_sense: Optional[int]
```

## Example Records

### XL-WSD

Supported languages (18): English (en), Arabic (ar), Bulgarian (bg), Catalan (ca), Czech (cs), Danish (da), German (de), Greek (el), Spanish (es), Estonian (et), Basque (eu), Finnish (fi), French (fr), Italian (it), Dutch (nl), Polish (pl), Portuguese (pt), Swedish (sv)

```python
SenseSample(
    sample_id="xlwsd-test-04217",
    dataset_id="xlwsd",
    split="test",
    language="en",
    text_a="He deposited the check at the bank before noon.",
    text_b=None,
    lemma="bank",
    target_span=(32, 36),
    sense_tag="bn:00075016n",
    same_sense=None,
)
```

### XL-WiC

Supported languages (12): English (en), Bulgarian (bg), German (de), Greek (el), Spanish (es), French (fr), Italian (it), Croatian (hr), Romanian (ro), Russian (ru), Turkish (tr), Chinese (zh)

```python
SenseSample(
    sample_id="xlwic-validation-1180",
    dataset_id="xlwic",
    split="validation",
    language="it",
    text_a="La barca è arrivata alla riva del lago.",
    text_b="La riva della strada era piena di fango.",
    lemma="riva",
    target_span=None,
    sense_tag=None,
    same_sense=0,
)
```

### MCL-WiC

Supported languages (46): Afrikaans (af), Amharic (am), Arabic (ar), Bulgarian (bg), Bengali (bn), German (de), Greek (el), English (en), Spanish (es), Estonian (et), Persian (fa), Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Croatian (hr), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Georgian (ka), Kazakh (kk), Korean (ko), Lithuanian (lt), Macedonian (mk), Malay (ms), Burmese (my), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Sinhala (si), Slovak (sk), Slovenian (sl), Swahili (sw), Tamil (ta), Thai (th), Turkish (tr), Ukrainian (uk), Urdu (ur), Vietnamese (vi), Xhosa (xh), Yoruba (yo), Chinese (zh), Zulu (zu)

```python
SenseSample(
    sample_id="mclwic-train-207",
    dataset_id="mclwic",
    split="train",
    language="tr",
    text_a="Çocuk ağaçtan düştü ama yaralanmadı.",
    text_b="Hisse senetlerinin değeri hızla düştü.",
    lemma="düşmek",
    target_span=None,
    sense_tag=None,
    same_sense=0,
)
```

Each record supplies enough context for both span-based probes (via `target_span` and `sense_tag`) and sentence-pair classification (via `text_b` and `same_sense`). Metrics can operate on a single iterator of `SenseSample`s without worrying about the original dataset format.
