# M3GLVQ — Modell-Versionen: was ist alt, was neu, was hat sich geändert

## Die 4 sauberen Dateien (kanonisch)

| Rolle | saubere Datei | Klasse | Repo-Ziel (Import) |
|---|---|---|---|
| Global **ALT** | `M3GLVQ_global_OLD.py` | `M3GLVQ`        | `src/m3glvq/M3GLVQ.py` |
| Global **NEU** | `M3GLVQ_global_NEW.py` | `M3GLVQ_Global` | `src/m3glvq/M3GLVQ_global.py` |
| Label  **ALT** | `M3GLVQ_label_OLD.py`  | `M3GLVQ_Label`  | `src/m3glvq/M3GLVQ_label_old.py` |
| Label  **NEU** | `M3GLVQ_label_NEW.py`  | `M3GLVQ_Label`  | `src/m3glvq/M3GLVQ_label.py` |

### Welche Uploads waren was (md5)
- `19f49e3c` → **Global ALT**  (M3GLVQ_3_.py)
- `7847d47f` → **Global NEU**  (M3GLVQ_global_2_/3_/5_.py — alle identisch)
- `1d1d975a` → Global *Zwischenstand* (M3GLVQ_global_old.py/_1_): korrigiert, aber **ohne** vektorisierten Scan — NICHT das echte Alt-Modell, nur langsamere Neu-Variante. Ignorieren.
- `752f9fd1` → **Label ALT**  (M3GLVQ_label_old.py/_1_)
- `490cd366` → **Label NEU**  (M3GLVQ_label_4_/5_/6_.py **und** das falsch benannte `M3GLVQ_label_old_2.py`)

---

## Änderungen GLOBAL (ALT → NEU) — echte Mathematik-Änderung

1. **Gewichts-Parametrisierung.** ALT: Simplex (`_project_simplex`, D* = Σ aᵥ Dᵥ, Gewichte auf Simplex, Summe 1). NEU: quadratisch (D* = Σ aᵥ² Dᵥ), Gradient mit Faktor **4·aᵥ**, **L2-Normierung** (`np.linalg.norm`). → WSOM-Gradientenkorrektur.
2. **Loss-Buchhaltung.** ALT: inkrementell `expected_new = loss[-1] + best_delta` → driftet, feuert `[Warning] Loss deviation`. NEU: exakter Recompute (Approach A), keine Drift.
3. **best-so-far.** NEU merkt niedrigsten je gesehenen Loss + zugehörige Gewichte/Prototypen und gibt *diese* zurück.
4. **Vektorisierter Kandidaten-Scan** `_scan_k` (Binärpfad) → ~2.8× schneller, bit-identisch.
5. **`get_vweights()`** liefert `a_sq`.

**Effekt:** unterscheidet sich bei **jedem** η (echte Parametrisierungs-Änderung). Beispiel K=6/η=0.02: Loss −176 → −339.

---

## Änderungen LABEL (ALT → NEU) — nur Speed + best-so-far

1. **`overall_L`-Caching.** ALT: `_assign()` rechnet die per-Label-Metrik intern jedes Mal neu (teuer, ~440×). NEU: `_assign(overall_L)` mit vorab gebauten `overall_L = [_overall_for_label(M,l) …]`. → ~1.9× schneller, **bit-identisch**.
2. **best-so-far** (`_track_best`) neu ergänzt.
3. Der **Weight-Update ist unverändert** und in BEIDEN bereits **zweiseitig** (`_weights_update_label(l, dp, dm, dp_V, dm_V, mask)` nutzt d⁺ **und** d⁻). Es gibt hier also KEINEN einseitig-Delta-Fix zwischen diesen beiden Dateien.

**Effekt — wichtig:** Bei kleinem/monotonem η (z. B. 0.01) ist NEU **bit-identisch** zu ALT — deshalb hast du `−233 / −233` gesehen. Der Unterschied erscheint **erst, wenn die Kurve nicht-monoton zurückdriftet** (höheres η, z. B. 0.03–0.05): dann hält best-so-far das Minimum, ALT driftet zurück. Für ein sichtbares Vorher/Nachher beim Label also **η hochdrehen**.

---

## Für die Lernkurven-Zelle
```python
from src.m3glvq.M3GLVQ            import M3GLVQ        as GlobalOld   # M3GLVQ_global_OLD.py
from src.m3glvq.M3GLVQ_global     import M3GLVQ_Global as GlobalNew   # M3GLVQ_global_NEW.py
from src.m3glvq.M3GLVQ_label_old  import M3GLVQ_Label  as LabelOld    # M3GLVQ_label_OLD.py
from src.m3glvq.M3GLVQ_label      import M3GLVQ_Label  as LabelNew    # M3GLVQ_label_NEW.py
```
Global zeigt den Unterschied bei jedem η; Label braucht η ≳ 0.03.
