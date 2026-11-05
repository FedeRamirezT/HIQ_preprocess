"""Detect spectral matches and create HTML/PNG galleries for local review.

Place beside all_components.h5 and template_low_freq_noisy_IC.h5, then:
    python detect_template_matches.py

Optional: --threshold 0.8 --input FILE --templates FILE --output DIRECTORY

Uses EXACT review_ICs transformation and positive Pearson r > threshold.
Each subplot overlays the candidate and best-matching template, both as
z-scored log10 PSD, with the template's family, source IC and correlation.
Selections are spectral matches, not confirmed artifacts. No EEG is changed.

All recordings are screened, including reserved subjects. Gallery sections
separate development, validation, and reference-source subjects. Matches in
reference-source subjects are descriptive, not independent validation. No
threshold is fitted. Template scores from the same subject are also exported
separately so in-sample matches cannot be mistaken for generalization.
"""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def normalize(psd):
    values = np.log10(np.asarray(psd, dtype=float) + np.finfo(float).eps)
    scale = values.std()
    if not np.isfinite(values).all() or not np.isfinite(scale) or scale == 0:
        return None
    return (values - values.mean()) / scale


def strings(dataset):
    return dataset.asstr()[:].tolist()


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=folder/'all_components.h5')
    parser.add_argument('--templates', type=Path, default=folder/'template_low_freq_noisy_IC.h5')
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--output', type=Path,
                        default=folder/('template_matches_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--split', type=Path, default=folder/'subject_split.json',
                        help='Optional frozen subject split; used to label gallery sections')
    args = parser.parse_args()
    if not -1 < args.threshold < 1:
        parser.error('--threshold must be between -1 and 1')
    if not args.input.is_file() or not args.templates.is_file():
        parser.error('Input or template H5 not found. Place both beside this script or pass their paths.')
    # Never overwrite an earlier review and its decisions.
    args.output.mkdir(parents=True, exist_ok=False)
    split = json.loads(args.split.read_text(encoding='utf-8')) if args.split.is_file() else None
    with h5py.File(args.templates, 'r') as bank:
        freqs = bank['freqs'][:]
        templates = bank['templates'][:]
        metadata = strings(bank['metadata']) if 'metadata' in bank else None
        template_runs = strings(bank['run']) if 'run' in bank else None
        families = strings(bank['family']) if 'family' in bank else [f'T{i+1:02d}' for i in range(len(templates))]
    if templates.ndim != 2 or templates.shape[1] != len(freqs) or not len(templates):
        raise ValueError('Invalid template dimensions')
    if not np.isfinite(templates).all() or np.any(templates.std(axis=1)==0):
        raise ValueError('Invalid or constant template')
    # Recenter and scale for Pearson calculation; this does not change r.
    template_z = (templates-templates.mean(axis=1,keepdims=True))/templates.std(axis=1,keepdims=True)
    source_subjects = [m[0] for m in metadata] if metadata is not None else ['']*len(templates)
    reference_subjects = set(source_subjects)-{''}
    rows, skipped, plot_data = [], [], {}
    with h5py.File(args.input,'r') as source:
        for key, group in source['recordings'].items():
            group_freqs = group['freqs'][:]
            subject = str(group.attrs['subject'])
            labels = json.loads(group['labels_json'].asstr()[()]) if 'labels_json' in group else {}
            partition = 'unassigned'
            if split:
                if subject in split['validation']:
                    partition = 'validation'
                elif subject in split['development']:
                    partition = 'development'
            if subject in reference_subjects:
                partition = 'reference_subject'
            for component, linear in zip(group['component_index'][:],group['psd_mean'][:]):
                identity = dict(group=key,subject=subject,session=str(group.attrs.get('session','')),
                                task=str(group.attrs.get('task','')),run=str(group.attrs.get('run','')),
                                component=int(component),partition=partition)
                if not np.isfinite(linear).all() or (linear<0).any():
                    skipped.append(dict(**identity,error='Invalid linear PSD')); continue
                if linear.shape != freqs.shape or not np.allclose(group_freqs,freqs):
                    if freqs[0]<group_freqs[0] or freqs[-1]>group_freqs[-1]:
                        skipped.append(dict(**identity,error='PSD does not cover template band')); continue
                    # Same interpolation order as review_ICs: linear PSD, before log.
                    linear = np.interp(freqs,group_freqs,linear)
                component_z = normalize(linear)
                if component_z is None:
                    skipped.append(dict(**identity,error='Constant or invalid transformed PSD')); continue
                correlations = np.clip(template_z@component_z/len(freqs),-1,1)
                best = int(np.argmax(correlations))
                matching = np.flatnonzero(correlations>args.threshold)
                foreign = [i for i,s in enumerate(source_subjects) if s and s!=subject]
                foreign_best = max(foreign,key=lambda i:correlations[i]) if foreign else None
                exact_reference = bool(metadata is not None and any(
                    subject==m[0] and identity['session']==m[1] and identity['task']==m[2]
                    and str(component)==m[3]
                    and (template_runs is None or identity['run']==template_runs[i])
                    for i,m in enumerate(metadata)))
                row = dict(**identity,selected=bool(len(matching)),best_family=families[best],
                           best_r=float(correlations[best]),best_template_index=best,
                           template_subject=source_subjects[best],
                           template_component=metadata[best][3] if metadata else '',
                           template_same_subject=source_subjects[best]==subject,
                           is_reference_component=exact_reference,
                           best_other_subject_family=families[foreign_best] if foreign_best is not None else '',
                           best_other_subject_r=float(correlations[foreign_best]) if foreign_best is not None else '',
                           matching_families=';'.join(families[i] for i in matching),
                           labels=';'.join(label for label,indices in labels.items() if int(component) in indices),
                           decision='')
                for i,value in enumerate(correlations):
                    row[f'r_{i+1:02d}_{families[i]}']=float(value)
                rows.append(row)
                if len(matching):
                    plot_data[(key,int(component))]=(component_z,best)
    rows.sort(key=lambda row:row['best_r'],reverse=True)
    selected = [row for row in rows if row['selected']]
    for number,row in enumerate(selected,1):
        row['match_id']=f'M{number:04d}'
    for filename,data in [('all_scores.csv',rows),('selected_components.csv',selected)]:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        if 'match_id' not in fields: fields.append('match_id')
        with (args.output/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(data)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(args.input.resolve()),
                templates=str(args.templates.resolve()),threshold=args.threshold,rule='positive Pearson r > threshold',
                scored=len(rows),selected=len(selected),skipped=skipped,
                split_loaded=bool(split),selection_warning='Spectral matches require human review; not confirmed artifacts')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if split:
        (args.output/'subject_split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    page_sections=[]
    for partition in ['development','validation','reference_subject','unassigned']:
        subset=[row for row in selected if row['partition']==partition]
        if not subset: continue
        pages=[]
        for start in range(0,len(subset),12):
            batch=subset[start:start+12]
            fig,axes=plt.subplots(4,3,figsize=(18,14),layout='constrained')
            for ax,row in zip(axes.flat,batch):
                component_z,best=plot_data[(row['group'],row['component'])]
                ax.plot(freqs,component_z,color='navy',lw=1.3,label='Component')
                ax.plot(freqs,template_z[best],color='darkorange',ls='--',lw=1.3,
                        label=f"Template {row['best_family']}")
                warning=' | SAME SUBJECT' if row['template_same_subject'] else ''
                if row['is_reference_component']: warning+=' | REFERENCE'
                ax.set_title(f"{row['match_id']} | {row['best_family']} | r={row['best_r']:.3f}{warning}\n"
                             f"sub-{row['subject']} IC {row['component']} ({row['labels']}) | "
                             f"ses-{row['session']} {row['task']} run-{row['run']}\n"
                             f"Template: sub-{row['template_subject']} IC {row['template_component']}",fontsize=9)
                ax.set(xlim=(float(freqs[0]),float(freqs[-1])),xlabel='Frequency (Hz)',ylabel='z-score of log10 PSD')
                ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
            for ax in list(axes.flat)[len(batch):]: ax.axis('off')
            fig.suptitle(f'Spectral matches | {partition} | r > {args.threshold:g} | zero-based IC indices\n'
                         'Each IC is shown once, overlaid with its best template; all matching families are in the CSV',fontsize=13)
            name=f'{partition}_{start//12+1:03d}.png'
            fig.savefig(args.output/name,dpi=120);plt.close(fig);pages.append(name)
        page_sections.append((partition,len(subset),pages))
    content='<!doctype html><html lang="es"><meta charset="utf-8"><title>Template matches</title>'
    content+='<style>body{font-family:system-ui;margin:24px}img{width:100%;max-width:2000px}a{margin-right:12px}</style>'
    content+=f'<h1>{len(selected)} coincidencias / {len(rows)} componentes evaluadas</h1>'
    content+=f'<p>Correlación de Pearson &gt; {args.threshold:g}. Azul: componente; naranja: template mejor correlacionado. '
    content+='Ambos espectros: log10 y z-score. Índices IC desde cero. Las coincidencias requieren revisión.</p>'
    content+='<p>REFERENCE indica una referencia original; SAME SUBJECT indica que el mejor template procede del mismo sujeto. '
    content+='No cuentan como validación independiente. El CSV incluye también la mejor correlación con referencias de otros sujetos.</p>'
    if not split:
        content+='<p>No se encontró subject_split.json: los sujetos sin referencia aparecen como unassigned.</p>'
    content+='<p><a href="selected_components.csv">Selección y decisiones</a><a href="all_scores.csv">Todas las correlaciones</a>'
    content+='<a href="summary.json">Resumen y errores</a></p>'
    for partition,count,pages in page_sections:
        content+=f'<h2>{html.escape(partition)}: {count} componentes</h2>'
        for name in pages:
            content+=f'<h3>{name}</h3><a href="{name}" target="_blank"><img loading="lazy" src="{name}"></a>'
    if not selected: content+='<p>No hay coincidencias con este umbral.</p>'
    content+='</html>'
    (args.output/'review.html').write_text(content,encoding='utf-8')
    print(f'{len(selected)} matches / {len(rows)} components; {len(skipped)} skipped')
    print(f'Open: {(args.output/"review.html").resolve()}')


if __name__=='__main__':
    main()
