# Private data interface

No patient images, masks, identifiers, manifests, or trained checkpoints are included in
this repository. The scripts expect deidentified local inputs with the interfaces below.

## Volume files

Each `.npz` volume contains:

- `image`: a three-dimensional sagittal MRI array in depth-height-width order.
- `mask`: a three-dimensional binary or integer lesion mask aligned to `image`.

Internal volume filenames follow `<deidentified_patient_id>_<sequence_id>.npz`, where
sequence ID `1` denotes T1-weighted MRI and `2` denotes T2-weighted MRI.

## Vision-language records

The Qwen scripts read a JSON list. Each record contains a `conversations` list with a user
message carrying an image path and lesion-coordinate prompt and an assistant message carrying
the infection/tumor answer. The exported records also include a four-element `bbox` in inclusive
`[x1, y1, x2, y2]` format and deidentified fields such as `patient_id`, `case_id`, `seq`, and
`slice_idx`. Hidden-box evaluation files preserve the record identity but replace the
reference-standard coordinate with a placeholder; predicted coordinates are inserted only at
inference time.

## Safety boundary

Never place source DICOM files, clinical identifiers, private ID mappings, absolute hospital
paths, or raw manifests in this repository. Use `.env.example` only as a template and keep the
actual `.env` file private.
