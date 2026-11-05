# Reproducir la revisión de ruido

Ejecutar con el entorno Python del proyecto, desde esta carpeta:

```powershell
python export_all_ic_psds.py
python review_noise_candidates.py
python detect_template_matches.py
```

El primer script regenera `all_components.h5` a partir de las ICA finales
existentes y de la configuración actual. Si esas ICA cambian, los PSD también.
El segundo crea la galería exploratoria en `candidates/review.html`.
El tercero compara con el template versionado de
`scripts/shared/templates/template_low_freq_noisy_IC.h5`, con umbral 0.9,
y crea `template_matches_FECHA_HORA/review.html`.

Conservar `subject_split.json` para mantener la división de sujetos.
Los PSD y las galerías son regenerables y se excluyen de Git.
No es necesaria una copia del template en esta carpeta.
